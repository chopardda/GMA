import os
import sys
import argparse

current_script_path = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_script_path)

sys.path.append(parent_directory)

from video_manager import VideoManager

# Create ArgumentParser object
parser = argparse.ArgumentParser(description='Show videos with tracked keypoints')
parser.add_argument('--tracked_kp_path', default='./output/tracked',
                    help='Path to tracked keypoints')
parser.add_argument("--missing_ok", default=False, action="store_true", help="Allow missing tracking info")

args = parser.parse_args()

video_folder = "/path/to/videos"

video_manager = VideoManager()
video_manager.add_all_videos(video_folder, add_pt_data=True)  # Load class data, not the videos themselves
video_manager.load_all_tracked_points(args.tracked_kp_path, missing_ok=args.missing_ok)
video_ids = video_manager.get_all_video_ids()

for video_id in video_ids:
    video_object = video_manager.get_video_object(video_id)
    video_object.show_video_with_tracks()
