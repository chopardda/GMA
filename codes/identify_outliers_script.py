import os
import sys
import argparse
import pickle
import numpy as np
from tqdm import tqdm

current_script_path = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_script_path)

sys.path.append(parent_directory)

from video_manager import VideoManager
from outlier_display import OutlierDisplay

# Create ArgumentParser object
parser = argparse.ArgumentParser(description='Check for outlier points in the tracked data')
# Add 'task' argument
parser.add_argument("--stddev_threshold", type=float, default=3, help="Number of standard deviations from"
                                                                      " the mean to consider a point an outlier")
parser.add_argument("--show_outliers", action="store_true", default=False, help="Visually show detected"
                                                                                "outliers")
parser.add_argument("--save_figures", action="store_true", default=False, help="Save figures of detected"
                                                                               "outliers")
parser.add_argument('--tracked_kp_path', default='./output/tracked',
                    help='Path to tracked keypoints')
parser.add_argument("--missing_ok", default=False, action="store_true", help="Allow missing tracking info")
parser.add_argument('--video_path',
                    default='/cluster/work/vogtlab/Projects/General_Movements/Preprocessed_Videos',
                    help='Path to directory containing videos.')

args = parser.parse_args()

video_folder = args.video_path

# Ensure outliers directory exists
if not os.path.exists("./output/outliers"):
    os.makedirs("./output/outliers")

video_manager = VideoManager()
video_manager.add_all_videos(video_folder, add_pt_data=True)  # Load class data, not the videos themselves
video_manager.load_all_tracked_points(args.tracked_kp_path, missing_ok=args.missing_ok, file_type='csv')
video_ids = video_manager.get_all_video_ids()

# Check if distribution of point movement is saved
if os.path.exists("./keypoint_distributions.pkl"):
    print("Loading existing distribution data")
    with open("./keypoint_distributions.pkl", "rb") as f:
        keypoint_distributions = pickle.load(f)

else:
    print("Gathering video statistics...")
    keypoint_frame_deltas = {}

    for video_id in video_ids:
        video_frame_deltas = video_manager.get_video_object(video_id).get_tracked_points_deltas()

        for keypoint, deltas in video_frame_deltas.items():
            if keypoint not in keypoint_frame_deltas:
                keypoint_frame_deltas[keypoint] = []
            keypoint_frame_deltas[keypoint] += deltas

    # Calculate distribution of point movement
    keypoint_distributions = {}

    for keypoint, deltas in keypoint_frame_deltas.items():
        # Process x and y coordinates of each keypoint
        x_deltas = [delta[0] for delta in deltas]
        y_deltas = [delta[1] for delta in deltas]

        # Calculate the mean and standard deviation of the x and y deltas
        x_mean = np.mean(x_deltas)
        x_std = np.std(x_deltas)
        y_mean = np.mean(y_deltas)
        y_std = np.std(y_deltas)

        # Save the results
        keypoint_distributions[keypoint] = {
            "x_mean": x_mean,
            "x_std": x_std,
            "y_mean": y_mean,
            "y_std": y_std
        }

        # Print the results
        print(f"Key point: {keypoint}")
        print(f"X mean: {x_mean}, X std: {x_std}")
        print(f"Y mean: {y_mean}, Y std: {y_std}")
        print("")

    # Save the keypoint_distributions
    with open("./keypoint_distributions.pkl", "wb") as f:
        pickle.dump(keypoint_distributions, f)


# Count the number of frames where the point moved more than the threshold * stddev from the mean
outlier_file = f"./output/outliers/outliers_{args.stddev_threshold}.pkl"
if os.path.exists(outlier_file):
    print("Loading existing outlier data")
    with open(outlier_file, "rb") as f:
        outliers = pickle.load(f)
        outlier_counts = [[len(x) for x in list(outliers[video_id].values())] for video_id in outliers]
        outlier_count = sum([sum(x) for x in outlier_counts])

else:
    outliers = {}
    outlier_count = 0

    for video_id in video_ids:
        video_object = video_manager.get_video_object(video_id)
        video_frame_deltas = video_object.get_tracked_points_deltas()
        outliers[video_id] = {}

        for keypoint, deltas in video_frame_deltas.items():
            x_deltas = [delta[0] for delta in deltas]
            y_deltas = [delta[1] for delta in deltas]

            x_mean = keypoint_distributions[keypoint]["x_mean"]
            x_std = keypoint_distributions[keypoint]["x_std"]
            y_mean = keypoint_distributions[keypoint]["y_mean"]
            y_std = keypoint_distributions[keypoint]["y_std"]

            x_diffs = np.abs(x_deltas - x_mean)
            y_diffs = np.abs(y_deltas - y_mean)
            x_outliers = np.where(x_diffs > args.stddev_threshold * x_std)[0]
            y_outliers = np.where(y_diffs > args.stddev_threshold * y_std)[0]

            outlier_set = set(x_outliers).union(set(y_outliers))

            if len(outlier_set) > 0:
                outliers[video_id][keypoint] = [
                    (i, x_diffs[i], x_diffs[i] / x_std, y_diffs[i],
                     y_diffs[i] / y_std) for i in
                    outlier_set]
                outlier_count += len(outlier_set)

        if len(outliers[video_id]) == 0:
            outliers.pop(video_id)

    # Save outliers
    with open(outlier_file, "wb") as f:
        pickle.dump(outliers, f)

print(f"Found {outlier_count} outliers")

if args.show_outliers and len(outliers) > 0:
    pbar = tqdm(total=outlier_count)
    for video_id in outliers:
        video_object = video_manager.get_video_object(video_id)

        od = OutlierDisplay(video_object, save_figures=args.save_figures, stddev_threshold=args.stddev_threshold)

        for keypoint, outlier_frame_info in outliers[video_id].items():
            for outlier_frame, x_diff, x_stddev_mul, y_diff, y_stddev_mul in outlier_frame_info:
                od.show_outlier(keypoint, outlier_frame, x_diff, x_stddev_mul, y_diff, y_stddev_mul)
                pbar.update(1)

        od.write_outliers_to_file()
        video_object.release_video()
