import os
import sys
import argparse
from tqdm import tqdm

current_script_path = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_script_path)

sys.path.append(parent_directory)

from video_manager import VideoManager
from point_relabeler import PointRelabeler

parser = argparse.ArgumentParser(description='Relabel detected outliers')
parser.add_argument('--outlier_dir', default='./output/outliers', help='Path to outlier data')
parser.add_argument('--tracked_kp_path', default='./output/tracked', help='Path to tracked keypoints')
parser.add_argument('--labeled_kp_path', default='./output/merged',
                    help='Path to directory containing merged labelled keypoints to use for cropping.')
parser.add_argument("--missing_ok", default=False, action="store_true", help="Allow missing tracking info")
parser.add_argument('--task', choices=['extreme_keypoints', 'all_body_keypoints'], required=True,
                    help='Task for labeling keypoints.')
parser.add_argument('--output_path', default='./output/relabeled',
                    help='Path to output directory for saving labelled data files.')
parser.add_argument('--video_path', default='/cluster/work/vogtlab/Projects/General_Movements/Preprocessed_Videos',
                    help='Path to folder containing videos.')

args = parser.parse_args()

video_folder = args.video_path
output_folder = args.output_path

if not os.path.exists(output_folder):
    os.makedirs(output_folder, exist_ok=True)

video_manager = VideoManager()
video_manager.add_all_videos(video_folder, add_pt_data=True)  # Load class data, not the videos themselves
video_manager.load_all_tracked_points(args.tracked_kp_path, missing_ok=args.missing_ok, file_type='csv')
video_manager.load_all_outlier_data(args.outlier_dir)
video_ids = video_manager.get_all_video_ids()

manual_correction_done = False

for video_id in tqdm(video_ids):
    video_object = video_manager.get_video_object(video_id)

    if video_object.outlier_data is not None and not video_object.keypoints_file_exists(args.output_path, args.task,
                                                                                        file_type='csv'):
        # Present user with the option to redefine the outlier points from the beginning to end of the video
        # For the rest of the points in that frame, provide the currently tracked points (these may optionally be
        # relabeled)
        outliers = video_object.get_sorted_outlier_table()
        relabeler = PointRelabeler()
        video_object.load_video()
        video_object.load_keypoint_labels_from_folder(args.labeled_kp_path, task=args.task, file_type='csv')

        # Handle the case where the outlier immediately returns to the correct place in the following frame
        one_offs = video_object.get_one_off_outliers()
        for idx, outlier in one_offs.iterrows():
            outlier_frame_index = outlier['Outlier frame index'] + 1
            outlier_keypoint = outlier['Keypoint']
            imp_x, imp_y = video_object.impute_value(outlier_keypoint, outlier_frame_index)
            adj_idx, adj_frame_idx = video_object.get_tracking_data_position(outlier_frame_index)
            video_object.tracking_data[adj_idx][outlier_keypoint][adj_frame_idx]['x'] = imp_x
            video_object.tracking_data[adj_idx][outlier_keypoint][adj_frame_idx]['y'] = imp_y
            video_object.update_arranged_tracked_data()

        # Remove oneoffs from outlier list
        outliers = outliers.drop(one_offs.index)

        # Store a list of already labelled frames, they don't need to be labeled more than once
        labelled_frames = []

        pbar = tqdm(total=len(outliers))
        for idx, outlier in outliers.iterrows():
            outlier_frame_index = outlier['Outlier frame index'] + 1

            if outlier['Outlier Type'] in [0, 3] or outlier['Outlier frame index'] + 1 in labelled_frames:
                continue

            labelled_frames.append(outlier_frame_index)
            point_set = video_object.get_point_set_for_outlier(outlier_frame_index)
            relabeler.relabel_points(video_object.video[outlier_frame_index], outlier_frame_index,
                                     point_set, outlier['Keypoint'], args.task)

            video_object.add_keypoint_labels(outlier_frame_index, relabeler.selected_points)
            pbar.update(1)

        pbar.close()

        # -- Save labelled points to file
        video_object.save_keypoints_to_csv(output_folder)
        video_object.save_keypoints_to_json(output_folder)

        # Save updated tracking data
        video_object.save_tracked_points_to_csv(args.tracked_kp_path)
        # video_object.save_tracked_points_to_json(args.tracked_kp_path)

        video_object.release_video()

if not manual_correction_done:
    print("No manual corrections were made, outliers have all been corrected")

else:
    print("Manual corrections have been made, please rerun the tracking script with the new labeled frames and check for outliers again")
