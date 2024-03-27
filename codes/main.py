CUDA_VISIBLE_DEVICES = 0  # TODO: use GPU

import os
import sys
import argparse

current_script_path = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_script_path)

sys.path.append(parent_directory)

from point_tracker import PointTracker
from video_manager import VideoManager


# TODO: all paths or as arguments!

def main():
    parser = argparse.ArgumentParser(description='Process videos.')
    parser.add_argument('--track_only', action='store_true',
                        help='Track keypoints only')
    parser.add_argument('--crop_resize_only', action='store_true',
                        help='Crop and resize videos only')
    parser.add_argument('--video_path',
                        default='/cluster/work/vogtlab/Projects/General_Movements/Preprocessed_Videos',
                        help='Path to directory containing videos.')
    parser.add_argument('--labeled_kp_path', default='./output/merged',
                        help='Path to directory containing merged labelled keypoints to use for cropping.')
    parser.add_argument('--tracked_kp_path', default='./output/tracked',
                        help='Path to output directory to save tracked keypoints later uses for cropping.')
    parser.add_argument('--cropped_videos_path', default='./output/cropped',
                        help='Path to output directory to save cropped videos.')
    parser.add_argument('--resized_videos_path', default='./output/resized',
                        help='Path to output directory to save cropped and resized videos.')

    args = parser.parse_args()

    video_folder = args.video_path
    labeled_keypoints_folder = args.labeled_kp_path
    tracked_keypoints_folder = args.tracked_kp_path
    cropped_videos_folder = args.cropped_videos_path
    resized_videos_folder = args.resized_videos_path

    # Ensure output directories exist
    if not os.path.exists(tracked_keypoints_folder):
        os.makedirs(tracked_keypoints_folder)

    if not os.path.exists(cropped_videos_folder):
        os.makedirs(cropped_videos_folder)

    if not os.path.exists(resized_videos_folder):
        os.makedirs(resized_videos_folder)

    video_manager = VideoManager()
    video_manager.add_all_videos(video_folder, add_pt_data=True)  # Load class data, not the videos themselves

    # Create point tracker
    tracker = PointTracker('../tapnet/checkpoints/tapir_checkpoint_panning.npy')

    video_ids = video_manager.get_all_video_ids()
    # video_ids = ['18_FN_c', '07_F-_c']

    # --- Track extreme coordinates for cropping
    if not args.crop_resize_only:
        for video_id in video_ids:
            # print(f"\n{'=' * 60}")
            print(f"Keypoint tracking for video {video_id}...")
            # print(f"{'=' * 60}\n")
            video_object = video_manager.get_video_object(video_id)

            # Define starting frame and tracking task (which keypoints to track)
            frame_index = 0
            task = 'extreme_keypoints'

            # Load video
            video_object.load_video()

            # Track points
            try:
                video_object.track_points(
                    tracker,
                    frame_index,
                    task,
                    labeled_keypoints_folder,
                    'json'
                )
            except ValueError as e:
                print(e)  # Handle the error appropriately

            # Save tracked points
            video_object.save_tracked_points_to_csv(tracked_keypoints_folder)
            video_object.save_tracked_points_to_json(tracked_keypoints_folder)

            video_object.release_video()

    # --- Crop videos according to tracked points
    if not args.track_only:
        for video_id in video_ids:

            print(f"Cropping and resizing video {video_id}...")
            video_object = video_manager.get_video_object(video_id)

            # -- Update extreme coordinates from tracked points (either already loaded or from files)
            # #- Option A: explicitly load the tracked point and update the coordinates
            # video_object.load_tracked_points_from_folder(tracked_keypoints_folder, 'json')
            # video_object.update_extreme_coordinates()

            # - Option B: implicitly load the tracked points when updating the coordinates.
            # Note: not needed if extreme_coordinated already loaded in video_object
            video_object.update_extreme_coordinates(tracked_keypoints_folder)
            print(video_object.extreme_coordinates)

            # -- Crop video and save to cropped_videos folder
            video_object.crop_and_resize_video(cropped_videos_folder, resize=True, resize_folder=resized_videos_folder,
                                               load_and_release_video=True)



    # # Now, iterate over the videos in the manager and process each one
    # for video_id in video_manager.video_collection:
    #     print(f"Processing video {video_id}: labelling")
    #     video_data = video_manager.get_video_data(video_id)
    #     video = video_data.video
    #
    #     if video_data is not None:
    #         # Instantiate PointLabeler
    #         point_labeler = PointLabeler(video_data)
    #
    #         # ToDo (optional): as long as some point is still (None, None) for all labeled frames, ask to label a frame at random
    #         desired_frame_indices = [0] #, video.metadata.num_images-1]  # first and last frame
    #
    #         # Label points at different timeframes
    #         for frame_index in desired_frame_indices:
    #             point_labeler.label_points(frame_index, task)
    #             video_manager.save_point_labels(video_id, frame_index, point_labeler.get_labels(frame_index))
    #
    #             point_label = video_manager.get_point_labels(video_id, frame_index)
    #             print(f'Video ID: {video_id}, frame: {frame_index}, point labels: {point_label}')

    # for video_id in video_manager.videos:
    #     frame_index = 0
    #     print(f'Processing video {video_id}: tracking and cropping')
    #     video_data = video_manager.get_video_data(video_id)
    #     video = video_data.video
    #
    #     point_label = video_manager.get_point_labels(video_id, frame_index)
    #
    #     # track the points and update extreme_coordinates
    #     video_manager.update_extreme_coordinates(video_id, tracker, video, frame_index, point_label)
    #     extreme_coords = video_manager.get_extreme_coordinates(video_id)
    #     if extreme_coords is not None:
    #         leftmost, topmost, rightmost, bottommost = extreme_coords
    #         print('extreme coords', leftmost, topmost, rightmost, bottommost)
    #
    #     tracks, visibles = video_manager.get_tracking_data(video_id, frame_index)
    #     print('saved tracks and visibles', tracks.shape, visibles.shape)
    #
    #     # Visualize tracks and save corresponding video
    #     video_viz = viz_utils.paint_point_track(video, tracks, visibles)
    #     fps_value = video.metadata.fps
    #     bps_value = video.metadata.bps
    #     file_path = os.path.join('output', f'tracked_points_{task}_video_{video_id}_frame_{frame_index}.mp4')
    #     print(f'wrote video {video_id} with tracked labels to', file_path)
    #     media.write_video(file_path, video_viz, fps=fps_value, bps=bps_value)
    #
    #     video_manager.crop_and_resize_video(video_id, 256, 256)

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
