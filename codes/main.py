CUDA_VISIBLE_DEVICES = 0 # TODO: use GPU

import os
import sys
import mediapy as media

current_script_path = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_script_path)

sys.path.append(parent_directory)

from point_labeler import PointLabeler
from point_tracker import PointTracker
from video_manager import VideoManager

from tapnet.utils import transforms
from tapnet.utils import viz_utils

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # 0 = all logs shown

# from keypoint_detector import KeypointDetector
# from video_cropper import VideoCropper
# from point_tracker import PointTracker
# from point_merger import PointMerger
# from feature_extractor import FeatureExtractor


def main():
    video_folder = "/home/daphne/Documents/GMA/data/video_few_samples"
    video_manager = VideoManager()

    # Process videos in sub-folders and extract metadata
    video_manager.add_all_videos(video_folder)

    # Create point tracker
    tracker = PointTracker('../tapnet/checkpoints/tapir_checkpoint_panning.npy')

    # Preprocessing
    task = 'extreme_keypoints'

    # Now, iterate over the videos in the manager and process each one
    for video_id in video_manager.videos:
        print(f"Processing video {video_id}: labelling")
        video_data = video_manager.get_video_data(video_id)
        video = video_data.video

        if video_data is not None:
            # Instantiate PointLabeler
            point_labeler = PointLabeler(video_data)

            # ToDo (optional): as long as some point is still (None, None) for all labeled frames, ask to label a frame at random
            desired_frame_indices = [0] #, video.metadata.num_images-1]  # first and last frame

            # Label points at different timeframes
            for frame_index in desired_frame_indices:
                point_labeler.label_points(frame_index, task)
                video_manager.save_point_labels(video_id, frame_index, point_labeler.get_labels(frame_index))

                point_label = video_manager.get_point_labels(video_id, frame_index)
                print(f'Video ID: {video_id}, frame: {frame_index}, point labels: {point_label}')


    for video_id in video_manager.videos:
        frame_index = 0
        print(f'Processing video {video_id}: tracking and cropping')
        video_data = video_manager.get_video_data(video_id)
        video = video_data.video

        point_label = video_manager.get_point_labels(video_id, frame_index)

        # track the points and update extreme_coordinates
        video_manager.update_extreme_coordinates(video_id, tracker, video, frame_index, point_label)
        extreme_coords = video_manager.get_extreme_coordinates(video_id)
        if extreme_coords is not None:
            leftmost, topmost, rightmost, bottommost = extreme_coords
            print('extreme coords', leftmost, topmost, rightmost, bottommost)

        tracks, visibles = video_manager.get_tracking_data(video_id, frame_index)
        print('saved tracks and visibles', tracks.shape, visibles.shape)

        # Visualize tracks and save corresponding video
        video_viz = viz_utils.paint_point_track(video, tracks, visibles)
        fps_value = video.metadata.fps
        bps_value = video.metadata.bps
        file_path = os.path.join('output', f'tracked_points_{task}_video_{video_id}_frame_{frame_index}.mp4')
        print(f'wrote video {video_id} with tracked labels to', file_path)
        media.write_video(file_path, video_viz, fps=fps_value, bps=bps_value)

        video_manager.crop_and_resize_video(video_id, 256, 256)

    # Preprocessing
    # keypoint tracing
    # features extraction

    # Further processing
    # merger = PointMerger()
    # extractor = FeatureExtractor()

    # Example: Process labeled, tracked, and merged points
    # features = extractor.extract_features(merged_points)

    # Save results, display them, or further analyze them


if __name__ == "__main__":
    main()
