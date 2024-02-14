import math
import mediapy as media
import numpy as np
import os

from PIL import Image


# TODO: reorganise and rename st that consistent with existing info in video_data.video.metadata
def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def find_extreme_coordinates(tracks, visibles):
    # Initialize extreme coordinates with first values
    x_coords = [pt[0] for pt in tracks[:, 0, :]]
    y_coords = [pt[1] for pt in tracks[:, 0, :]]

    topmost, bottommost = max(y_coords), min(y_coords)  # get top- and bottom-most selected points
    rightmost, leftmost = max(x_coords), min(x_coords)  # get right- and left-most selected points

    num_keypoints, num_frames, _ = tracks.shape

    for keypoint in range(num_keypoints):
        for frame in range(num_frames):
            if visibles[keypoint, frame]:
                x, y = tracks[keypoint, frame]

                topmost = max(topmost, math.ceil(y))
                bottommost = min(bottommost, math.floor(y))
                leftmost = min(leftmost, math.floor(x))
                rightmost = max(rightmost, math.ceil(x))

    return leftmost, topmost, rightmost, bottommost


class VideoManager:
    def __init__(self):
        # Stores both video data and metadata
        self.videos = {}  # Format: { video_id: { "data": VideoData, "metadata": VideoMetadata }, ... }
        self.point_labels = {}  # Stores point labels for each video {video_id: {frame_index: labels}}
        self.tracking_data = {}  # Stores tracking data {video_id: {frame_index: (tracks, visibles)}} #TODO: change tracks, visible to dict
        self.extreme_coordinates = {}  # Format: {video_id: (leftmost, topmost, rightmost, bottommost)}

    def add_all_videos(self, folder):
        categories = {
            'AI_GeMo_early_N': {'age_group': 'younger', 'health_status': 'normal'},
            'AI_GeMo_early_PR': {'age_group': 'younger', 'health_status': 'abnormal'},
            'AI_GeMo_late_FN': {'age_group': 'older', 'health_status': 'normal'},
            'AI_GeMo_late_F-': {'age_group': 'older', 'health_status': 'abnormal'}
        }

        for category, info in categories.items():
            print(category)
            subfolder_path = os.path.join(folder, category)
            print(subfolder_path)
            for filename in os.listdir(subfolder_path):
                print(filename)
                if filename.lower().endswith('.mp4'):  # Add other video formats if needed
                    video_id = filename.split('_')[0]
                    file_path = os.path.join(subfolder_path, filename)
                    print(file_path)
                    video_metadata = VideoMetadata(video_id,
                                                   file_path,
                                                   info['age_group'],
                                                   info['health_status'])

                    video = media.read_video(file_path)
                    video_data = VideoData(video)

                    self.videos[video_id] = {"data": video_data, "metadata": video_metadata}
            break  # TODO: remove break (only loading first category for development)

    # def add_video(self, file_path):
    #     
    #     video_data = VideoData(video_array, height, width)
    #     video_metadata = VideoMetadata(video_id, file_path, age_group, health_status)
    #     self.videos[video_id] = {"data": video_data, "metadata": video_metadata}

    def get_video_data(self, video_id):
        video = self.videos.get(video_id, None)
        return video["data"] if video else None

    def get_video_metadata(self, video_id):
        video = self.videos.get(video_id, None)
        return video["metadata"] if video else None

    def remove_video(self, video_id):
        if video_id in self.videos:
            del self.videos[video_id]

    def show_video(self, video_id):
        vid = self.videos.get(video_id, None)
        media.show_video(vid.video, fps=vid.fps)

    def save_point_labels(self, video_id, frame_index, labels):
        if video_id not in self.point_labels:
            self.point_labels[video_id] = {}
        self.point_labels[video_id][frame_index] = labels

    def get_point_labels(self, video_id, frame_index):
        return self.point_labels.get(video_id, {}).get(frame_index, None)

    def save_tracking_data(self, video_id, frame_index, tracks, visibles):
        if video_id not in self.tracking_data:
            self.tracking_data[video_id] = {}
        self.tracking_data[video_id][frame_index] = (tracks, visibles)

    def get_tracking_data(self, video_id, frame_index):
        return self.tracking_data.get(video_id, {}).get(frame_index, (None, None))

    def update_extreme_coordinates(self, video_id, tracker, video, frame_index, point_label):
        tracks, visibles = tracker.track_selected_points(video, frame_index, point_label)
        self.save_tracking_data(video_id, frame_index, tracks, visibles)
        extreme_coords = find_extreme_coordinates(tracks, visibles)
        self.extreme_coordinates[video_id] = extreme_coords

    def get_extreme_coordinates(self, video_id):
        return self.extreme_coordinates.get(video_id, None)

    def crop_and_resize_video(self, video_id, resize_height, resize_width):
        video = self.get_video_data(video_id).video
        height, width = video.metadata.shape
        fps_value = video.metadata.fps
        bps_value = video.metadata.bps

        x_min, y_max, x_max, y_min = self.get_extreme_coordinates(video_id)

        # Get spread of selected points
        x_spread = x_max - x_min
        y_spread = y_max - y_min

        # Crop around selected points
        margin = 0.15  # 15% margin
        x_margin = round(margin * x_spread)
        y_margin = round(margin * y_spread)
        largest_margin = max(x_margin, y_margin)

        min_x_crop, max_x_crop = (max(0, x_min - largest_margin - 1),
                                  min(width, x_max + largest_margin + 1))
        min_y_crop, max_y_crop = (max(0, y_min - largest_margin - 1),
                                  min(height, y_max + largest_margin + 1))

        cropped_width = max_x_crop - min_x_crop
        cropped_height = max_y_crop - min_y_crop

        frames_original = video.__array__()
        frames_cropped = frames_original[:, min_y_crop:max_y_crop, min_x_crop:max_x_crop, :]

        # Expand to square (background padding) and resize to desired dimensions
        frames_crop_squared = []
        frames_crop_resized = []
        for frame in frames_cropped:
            im = Image.fromarray(frame, mode="RGB")
            squared_frame = expand2square(im, (255, 255, 255))
            resized_frame = squared_frame.resize((resize_height, resize_width))
            frames_crop_squared.append(np.asarray(squared_frame))
            frames_crop_resized.append(np.asarray(resized_frame))

        media.write_video(
            os.path.join('output', 'cropped_videos', f'cropped_vid_{video_id}.mp4'),
            frames_cropped,
            fps=fps_value,
            bps=bps_value
        )
        print('wrote cropped video to', os.path.join('output', 'cropped_videos', f'cropped_vid_{video_id}.mp4'))

        media.write_video(
            os.path.join('output', 'cropped_videos', f'cropped_resized_vid_{video_id}.mp4'),
            frames_crop_resized,
            fps=fps_value,
            bps=bps_value
        )
        print('wrote cropped and resized video to',
              os.path.join('output', 'cropped_videos', f'cropped_resized_vid_{video_id}.mp4'))


class VideoData:
    def __init__(self, video, fps=30):
        self.video = video
        self.height, self.width = video.shape[1:3]
        self.n_frames = video.metadata.num_images
        self.fps = video.metadata.fps


class VideoMetadata:
    def __init__(self, video_id, file_path, age_group, health_status):
        self.video_id = video_id
        self.file_path = file_path
        self.age_group = age_group
        self.health_status = health_status
