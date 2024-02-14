import haiku as hk
import jax
import mediapy as media
import numpy as np
import tree

from tapnet import tapir_model
from tapnet.utils import transforms


def build_model(frames, query_points):
    """Compute point tracks and occlusions given frames and query points."""
    model = tapir_model.TAPIR(bilinear_interp_with_depthwise_conv=False, pyramid_level=0)
    outputs = model(
        video=frames,
        is_training=False,
        query_points=query_points,
        query_chunk_size=64,
    )
    return outputs


def preprocess_frames(frames):
    """Preprocess frames to model inputs.

    Args:
      frames: [num_frames, height, width, 3], [0, 255], np.uint8

    Returns:
      frames: [num_frames, height, width, 3], [-1, 1], np.float32
    """
    frames = frames.astype(np.float32)
    frames = frames / 255 * 2 - 1
    return frames


def postprocess_occlusions(occlusions, expected_dist):
    """Postprocess occlusions to boolean visible flag.

    Args:
      occlusions: [num_points, num_frames], [-inf, inf], np.float32
      expected_dist: [num_points, num_frames], [-inf, inf], np.float32

    Returns:
      visibles: [num_points, num_frames], bool
    """
    visibles = (1 - jax.nn.sigmoid(occlusions)) * (1 - jax.nn.sigmoid(expected_dist)) > 0.5
    return visibles


def sample_random_points(frame_max_idx, height, width, num_points):
    """Sample random points with (time, height, width) order."""
    y = np.random.randint(0, height, (num_points, 1))
    x = np.random.randint(0, width, (num_points, 1))
    t = np.random.randint(0, frame_max_idx + 1, (num_points, 1))
    points = np.concatenate((t, y, x), axis=-1).astype(np.int32)  # [num_points, 3]
    return points


# def convert_select_points_to_query_points(frame, points):
#     """Convert select points to query points.
#
#     Args:
#       points: [num_points, 2], in [x, y]
#     Returns:
#       query_points: [num_points, 3], in [t, y, x]
#     """
#     points = np.stack(points)
#     query_points = np.zeros(shape=(points.shape[0], 3), dtype=np.float32)
#     query_points[:, 0] = frame
#     query_points[:, 1] = points[:, 1]
#     query_points[:, 2] = points[:, 0]
#     return query_points


def convert_select_point_dict_to_query_points(frame, points_dict):
    """
    Convert select points to query points.

    Args:
      frame: The frame number.
      points_dict: Dictionary with keypoints as keys and [x, y] as values.
    Returns:
      query_points: [num_points, 3], in [t, y, x] format.
    """
    # Extract the values (coordinates) from the dictionary and convert to a numpy array
    points = np.array(list(points_dict.values()))
    query_points = np.zeros(shape=(points.shape[0], 3), dtype=np.float32)
    query_points[:, 0] = frame
    query_points[:, 1] = points[:, 1]  # y-coordinates
    query_points[:, 2] = points[:, 0]  # x-coordinates

    return query_points


class PointTracker:
    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        self.resize_height, self.resize_width = 256, 256
        self.model, self.params, self.state = self.initialize_model()

    def initialize_model(self):
        ckpt_state = np.load(self.checkpoint_path, allow_pickle=True).item()
        params, state = ckpt_state['params'], ckpt_state['state']
        model = hk.transform_with_state(build_model)
        return jax.jit(model.apply), params, state

    def inference(self, frames, query_points):
        """Inference on one video.

        Args:
          frames: [num_frames, height, width, 3], [0, 255], np.uint8
          query_points: [num_points, 3], [0, num_frames/height/width], [t, y, x]

        Returns:
          tracks: [num_points, 3], [-1, 1], [t, y, x]
          visibles: [num_points, num_frames], bool
        """
        # Preprocess video to match model inputs format
        frames = preprocess_frames(frames)
        # num_frames, height, width = frames.shape[0:3]
        query_points = query_points.astype(np.float32)
        frames, query_points = frames[None], query_points[None]  # Add batch dimension

        # Model inference
        rng = jax.random.PRNGKey(42)
        outputs, _ = self.model(self.params, self.state, rng, frames, query_points)
        outputs = tree.map_structure(lambda x: np.array(x[0]), outputs)
        tracks, occlusions, expected_dist = outputs['tracks'], outputs['occlusion'], outputs['expected_dist']

        # Binarize occlusions
        visibles = postprocess_occlusions(occlusions, expected_dist)
        return tracks, visibles

    def track_random_points(self, video, num_points):
        frames = media.resize_video(video, (self.resize_height, self.resize_width))
        query_points = sample_random_points(0, frames.shape[1], frames.shape[2], num_points)

        tracks, visibles = self.inference(frames, query_points)
        return tracks, visibles

    def track_selected_points(self, video, select_frame, select_points, track_backward=False):
        height, width = video.metadata.shape
        frames = media.resize_video(video, (self.resize_height, self.resize_width)) # shape (n_frames, h, w, 3)

        if track_backward:
            frames = frames[:select_frame + 1][::-1]  # Reverse the frames up to the selected frame
        else:
            frames = frames[select_frame:]  # Start from the selected frame to the end

        # Convert from (x,y) to (t,y,x), where t = select_frame
        query_points = convert_select_point_dict_to_query_points(select_frame, select_points)

        # From original image size to resized dimensions
        query_points = transforms.convert_grid_coordinates(
            query_points, (1, height, width), (1, self.resize_height, self.resize_width),
            coordinate_format='tyx')

        # Tracks has shape (n_keypoints, n_frames, 2) and visibles has shape (n_keypoints, n_frames)
        tracks, visibles = self.inference(frames, query_points)

        # Reverse the tracks to match the original video order
        if track_backward:
            tracks = tracks[::-1]
            visibles = visibles[::-1]

        # From resized dimensions back to original image size
        tracks = transforms.convert_grid_coordinates(tracks, (self.resize_height, self.resize_width),
                                                     (width, height))

        return tracks, visibles

    def merge_tracks(self, forward_tracks, backward_tracks, visibles_forward, visibles_backward, threshold):
        num_keypoints, num_frames, _ = forward_tracks.shape
        merged_tracks = np.zeros_like(forward_tracks)
        merged_visibles = np.zeros((num_keypoints, num_frames), dtype=bool)

        for keypoint in range(num_keypoints):
            for frame in range(num_frames):
                point_forward = forward_tracks[keypoint, frame]
                point_backward = backward_tracks[keypoint, frame]
                visible_forward = visibles_forward[keypoint, frame]
                visible_backward = visibles_backward[keypoint, frame]

                if visible_forward and visible_backward:
                    distance = np.linalg.norm(point_forward - point_backward)
                    if distance < threshold:
                        # Merge points by averaging
                        merged_tracks[keypoint, frame] = (point_forward + point_backward) / 2
                        merged_visibles[keypoint, frame] = True
                    else:
                        # Handle points that are not close enough
                        # For simplicity, choose the point from forward tracking
                        merged_tracks[keypoint, frame] = point_forward
                        merged_visibles[keypoint, frame] = visible_forward
                elif visible_forward:
                    merged_tracks[keypoint, frame] = point_forward
                    merged_visibles[keypoint, frame] = True
                elif visible_backward:
                    merged_tracks[keypoint, frame] = point_backward
                    merged_visibles[keypoint, frame] = True
                # If neither are visible, the point remains as zero and visible as False

        return merged_tracks, merged_visibles