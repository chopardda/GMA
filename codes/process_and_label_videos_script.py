import os
import sys
import argparse
import datetime

current_script_path = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_script_path)

sys.path.append(parent_directory)

from video_manager import VideoManager

# Create ArgumentParser object
parser = argparse.ArgumentParser(description='Process and label videos.')
# Add 'task' argument
parser.add_argument('--task', choices=['extreme_keypoints', 'all_body_keypoints'], required=True,
                    help='Task for labeling keypoints.')
parser.add_argument('--tag', default=f"{os.environ.get('USER')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    help='Tag for the output files.')
parser.add_argument('--relabel', default=False,
                    help='Specifies whether to relabel keypoints in videos that are already labeled. '
                         'If set, videos with existing labels will be labeled one more time with a different filename. '
                         'By default, existing labels in videos are preserved and not replaced.')
parser.add_argument('--video_path', default='/cluster/work/vogtlab/Projects/General_Movements/Preprocessed_Videos',
                    help='Path to folder containing videos.')
parser.add_argument('--output_path', default='./output/labeled',
                    help='Path to folder containing videos.')
args = parser.parse_args()

video_folder = args.video_path
output_folder = args.output_path

if not os.path.exists(output_folder):
    os.makedirs(output_folder, exist_ok=True)

video_manager = VideoManager()
video_manager.add_all_videos(video_folder, add_pt_data=True)  # Load class data, not the videos themselves

video_ids = video_manager.get_all_video_ids()
# video_ids = ['18_FN_c', '07_F-_c']

for video_id in video_ids:
    print(f"Labelling process for video {video_id}...")

    # -- If labels already present, skip video
    if not args.relabel:
        prefix = f"{video_id}.{args.task}"
        file_exists = any(fname.startswith(prefix) for fname in os.listdir(output_folder))
        if file_exists:
            print(f"A file starting with {prefix} exists in {output_folder}. "
                  f"Video {video_id} is being skipped.")
            continue

    # -- Get video object
    video_object = video_manager.get_video_object(video_id)

    # -- Load video
    video_object.load_video()

    # -- Label video
    frame_index = 0
    video_object.label_and_store_keypoints(frame_index, task=args.task)

    # -- Save labelled points to file
    video_object.save_keypoints_to_csv(output_folder, args.tag)
    video_object.save_keypoints_to_json(output_folder, args.tag)

    # -- Release video from memory
    video_object.release_video()
