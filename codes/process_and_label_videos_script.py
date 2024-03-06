import os
import sys
import argparse

current_script_path = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_script_path)

sys.path.append(parent_directory)

from video_manager import VideoManager

# Create ArgumentParser object
parser = argparse.ArgumentParser(description='Process and label videos.')
# Add 'task' argument
parser.add_argument('--task', choices=['extreme_keypoints', 'all_body_keypoints'], required=True,
                    help='Task for labeling keypoints.')

args = parser.parse_args()

video_folder = "/cluster/work/vogtlab/Projects/General_Movements/Preprocessed_Videos"
output_folder = "./output/labeled"

if not os.path.exists(output_folder):
    os.makedirs(output_folder, exist_ok=True)

video_manager = VideoManager()
video_manager.add_all_videos(video_folder, add_pt_data=True)  # Load class data, not the videos themselves

video_ids = video_manager.get_all_video_ids()
# video_ids = ['18_FN_c', '07_F-_c']

for video_id in video_ids:
    print(f"Labelling process for video {video_id}...")
    video_object = video_manager.get_video_object(video_id)

    # -- Load video
    video_object.load_video()

    # -- Label video
    frame_index = 0
    video_object.label_and_store_keypoints(frame_index, task=args.task)

    # -- Save labelled points to file
    video_object.save_keypoints_to_csv(output_folder)
    video_object.save_keypoints_to_json(output_folder)

    # -- Release video from memory
    video_object.release_video()
