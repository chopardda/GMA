import os
import sys
import argparse

current_script_path = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_script_path)

sys.path.append(parent_directory)

from video_manager import VideoManager

parser = argparse.ArgumentParser(description='Relabel detected outliers')
parser.add_argument('--outlier_dir', default='./output/outliers', help='Path to outlier data')
parser.add_argument('--tracked_kp_path', default='./output/tracked', help='Path to tracked keypoints')
parser.add_argument('--labeled_kp_path', default='./output/merged',
                    help='Path to directory containing merged labelled keypoints to use for cropping.')
parser.add_argument("--missing_ok", default=False, action="store_true", help="Allow missing tracking info")

args = parser.parse_args()

video_folder = "/cluster/work/vogtlab/Projects/General_Movements/Preprocessed_Videos"

video_manager = VideoManager()
video_manager.add_all_videos(video_folder, add_pt_data=True)  # Load class data, not the videos themselves
video_manager.load_all_tracked_points(args.tracked_kp_path, missing_ok=args.missing_ok)
video_manager.load_all_outlier_data(args.outlier_dir)
video_ids = video_manager.get_all_video_ids()

for video_id in video_ids:
    video_object = video_manager.get_video_object(video_id)

    if video_object.outlier_data is not None:
        # Present user with the option to redefine the outlier points from the beginning to end of the video
        # For the rest of the points in that frame, provide the currently tracked points (these may optionally be
        # relabeled)
        outliers = video_object.get_sorted_outlier_table()

        for idx, outlier in outliers.iterrows():
            video_object.load_keypoint_labels_from_folder(args.labeled_kp_path, )
            point_set = video_object.get_point_set_for_outlier(outlier['Frame index'], outlier['Keypoint'],
                                                               outlier['Outlier frame index'])
