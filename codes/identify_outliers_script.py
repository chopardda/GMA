import os
import sys
import argparse
import numpy as np

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

args = parser.parse_args()

video_folder = "/cluster/work/vogtlab/Projects/General_Movements/Preprocessed_Videos"
tracked_folder = "./output/tracked"

video_manager = VideoManager()
video_manager.add_all_videos(video_folder, add_pt_data=True)  # Load class data, not the videos themselves
video_manager.load_all_tracked_points(tracked_folder)
video_ids = video_manager.get_all_video_ids()

print("Gathering video statistics...")
frame_index = 0  # Use this later when we have more index frames
keypoint_frame_deltas = {}

for video_id in video_ids:
    video_frame_deltas = video_manager.get_video_object(video_id).get_tracked_points_deltas(frame_index)

    for keypoint, deltas in video_frame_deltas.items():
        if keypoint not in keypoint_frame_deltas:
            keypoint_frame_deltas[keypoint] = []
        keypoint_frame_deltas[keypoint].append(deltas)

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

# Count the number of frames where the point moved more than the threshold * stddev from the mean
outliers = {}
outlier_count = 0

for video_id in video_ids:
    video_object = video_manager.get_video_object(video_id)
    video_frame_deltas = video_object.get_tracked_points_deltas(frame_index)
    outliers[video_id] = {}
    outliers[video_id][frame_index] = {}

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
        outliers[video_id][frame_index][keypoint] = [
            (i, x_diffs[i], x_diffs[i] / x_std, y_diffs[i],
             y_diffs[i] / y_std) for i in
            outlier_set]
        outlier_count += len(outlier_set)

print(f"Found {outlier_count} outliers")

if args.show_outliers and len(outliers) > 0:
    for video_id in outliers:
        video_object = video_manager.get_video_object(video_id)
        video_object.load_video()

        od = OutlierDisplay(video_object, save_figures=args.save_figures, stddev_threshold=args.stddev_threshold)

        for frame_index in outliers[video_id]:
            for keypoint, outlier_frame_info in outliers[video_id][frame_index].items():
                for outlier_frame, x_diff, x_stddev_mul, y_diff, y_stddev_mul in outlier_frame_info:
                    od.show_outlier(keypoint, frame_index, outlier_frame, x_diff, x_stddev_mul, y_diff, y_stddev_mul)

        od.write_outliers_to_file()
        video_object.release_video()
