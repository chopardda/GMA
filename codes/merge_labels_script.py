import os
import sys
import argparse
import datetime

current_script_path = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_script_path)

sys.path.append(parent_directory)

from video_manager import VideoManager

# Create ArgumentParser object
parser = argparse.ArgumentParser(description='Merge point label sets.')
# Add 'task' argument
parser.add_argument('--task', choices=['extreme_keypoints', 'all_body_keypoints'], required=True,
                    help='Task for labeling keypoints.')

args = parser.parse_args()

video_folder = "/cluster/work/vogtlab/Projects/General_Movements/Preprocessed_Videos"
labeled_folder = "./output/labeled"
merged_folder = "./output/merged"

if not os.path.exists(merged_folder):
    os.makedirs(merged_folder, exist_ok=True)

video_manager = VideoManager()
video_manager.add_all_videos(video_folder, add_pt_data=True)  # Load class data, not the videos themselves

video_ids = video_manager.get_all_video_ids()

for video_id in video_ids:
    print(f"Merging labels for video {video_id}...")
    video_object = video_manager.get_video_object(video_id)

    # -- Load video
    video_object.load_video()

    # -- Load labeled data
    label_sets = video_object.load_labeled_keypoint_sets(labeled_folder, args.task)

    print(f"Found {len(label_sets)} label sets")

    # -- Merge each labeled frame set
    video_object.merge_points(label_sets, 0, args.task)

    # -- Save merged points to file
    video_object.save_keypoints_to_csv(merged_folder)
    video_object.save_keypoints_to_json(merged_folder)

    # -- Release video from memory
    video_object.release_video()