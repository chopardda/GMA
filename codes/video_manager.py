import csv
import json
import math
import mediapy as media
import numpy as np
import os
import glob
import seaborn as sns

from collections import defaultdict
from PIL import Image
import pandas as pd
from point_labeler import PointLabeler
from point_merger import PointMerger


# TODO: move static functions to utils
def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def find_extreme_coordinates(tracking_data):
    # Initialize extreme coordinates
    topmost = bottommost = leftmost = rightmost = None

    # Iterate through each frame
    for frame_index, keypoints in tracking_data.items():
        # Iterate through each keypoint in the frame
        for keypoint, data_list in keypoints.items():
            for data in data_list:
                if data['visible']:
                    x, y = data['x'], data['y']

                    if topmost is None or y > topmost:
                        topmost = y
                    if bottommost is None or y < bottommost:
                        bottommost = y
                    if leftmost is None or x < leftmost:
                        leftmost = x
                    if rightmost is None or x > rightmost:
                        rightmost = x

    # Apply rounding or flooring as needed to ensure the coordinates are integers
    topmost = int(topmost) if topmost is not None else None
    bottommost = int(bottommost) if bottommost is not None else None
    leftmost = int(leftmost) if leftmost is not None else None
    rightmost = int(rightmost) if rightmost is not None else None

    return {'leftmost': leftmost, 'topmost': topmost, 'rightmost': rightmost, 'bottommost': bottommost}


def get_patient_data_from_filename(filename):
    """
    Get age group and health status based on video filename

    Args:
        filename (str): Name of the video file.
    """

    categories = {
        'N': {'age_group': 'younger', 'health_status': 'normal'},
        'PR': {'age_group': 'younger', 'health_status': 'abnormal'},
        'FN': {'age_group': 'older', 'health_status': 'normal'},
        'F-': {'age_group': 'older', 'health_status': 'abnormal'}
    }

    group = os.path.splitext(filename)[0].split('_')[1]
    age_group = categories[group]['age_group'] if group in categories else None
    health_status = categories[group]['health_status'] if group in categories else None

    return age_group, health_status


#################################################
############# VideoManager Class ################
#################################################

class VideoManager:
    def __init__(self):
        # Stores videos
        self.video_collection = {}  # Format: { video_id: VideoObject, ... }

    def add_video(self, filepath, video_size, load_now=False, add_pt_data=False):
        filename = os.path.basename(filepath)
        video_id = os.path.splitext(filename)[0]

        if load_now:
            video = media.read_video(filepath)
        else:
            video = None

        if add_pt_data:
            age_group, health_status = get_patient_data_from_filename(filename)
            patient_data = PatientData(age_group, health_status)
        else:
            patient_data = None

        self.video_collection[video_id] = VideoObject(filepath, video_id, video_size, video, patient_data)

    def add_all_videos(self, folder, load_now=False, add_pt_data=False):
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.lower().endswith('.mp4'):
                    full_filename = os.path.join(root, file)
                    self.add_video(full_filename, os.path.getsize(full_filename), load_now, add_pt_data)

    def get_all_video_ids(self, sort=True):
        if sort:
            return sorted(list(self.video_collection.keys()), key=lambda x: int(self.video_collection[x].video_size))

        return list(self.video_collection.keys())

    def get_video_object(self, video_id):
        """
        Get the VideoObject instance for the given video ID.

        Returns:
            VideoObject: The VideoObject instance for the given video ID, or None if no such video exists.
        """
        video_object = self.video_collection.get(video_id, None)
        return video_object if video_object else None

    def remove_video(self, video_id):
        if video_id in self.video_collection:
            del self.video_collection[video_id].video

    def load_all_tracked_points(self, tracked_keypoints_folder, file_type='json', missing_ok=False):
        video_keys = list(self.video_collection.keys())
        for video_id in video_keys:
            try:
                self.video_collection[video_id].load_tracked_points_from_folder(tracked_keypoints_folder, file_type)

            except FileNotFoundError as e:
                if not missing_ok:
                    raise FileNotFoundError(f"Error loading tracked points for video {video_id}: {e}")

                else:
                    # Remove video from collection
                    self.video_collection.pop(video_id)

    def load_all_outlier_data(self, outlier_data_folder):
        video_keys = list(self.video_collection.keys())
        for video_id in video_keys:
            self.video_collection[video_id].load_outlier_data_from_folder(outlier_data_folder)


#################################################
############# VideoObject Class ################
#################################################

class VideoObject:
    def __init__(self, file_path, video_id, video_size, video=None, patient_data=None):
        self.file_path = file_path
        self.video_id = video_id
        self.video = video  # a VideoArray object
        self.video_size = video_size

        self.patient_data = patient_data

        self.keypoint_labels = {}  # Stores point labels for each video {frame_index: labels, ...}
        self.tracking_data = {}  # Stores tracked points {'point': [{'x': x_val, 'y':y_val, 'visible': bool), ...], ...}
        self.arranged_tracking_data = {}
        self.extreme_coordinates = {}  # Format: {leftmost: , topmost: , rightmost: , bottommost: }

        self.labeling_task = None

        self.outlier_data = None
        self.colour_map = VideoObject._get_colour_map()

    @staticmethod
    def _get_colour_map():
        full_set_body_keypoints = ['nose',
                                   'head bottom', 'head top',
                                   'left ear', 'right ear',
                                   'left shoulder', 'right shoulder',
                                   'left elbow', 'right elbow',
                                   'left wrist', 'right wrist',
                                   'left hip', 'right hip',
                                   'left knee', 'right knee',
                                   'left ankle', 'right ankle']
        # Use a matplotlib colormap
        colorpalette = sns.color_palette("hls",
                                         len(full_set_body_keypoints))  # 'tab20' is a good palette for distinct colors

        for i in range(len(colorpalette)):
            colorpalette[i] = tuple([int(255 * x) for x in colorpalette[i]])

        bodypart_colors = {
            full_set_body_keypoints[i]: colorpalette[i] for i in range(len(full_set_body_keypoints))
        }

        return bodypart_colors

    def load_video(self):
        """
        Loads the video in the VideoObject if there is no video loaded yet.
        """
        if self.video is None:
            # print(f'Loading video {self.video_id}')
            self.video = media.read_video(self.file_path)

    def release_video(self):
        """
        Release the video in the VideoObject if loaded.
        """
        if self.video is not None:
            # print(f'Releasing video {self.video_id}')
            self.video = None

    def show_video(self):
        """
        Displays the video currently loaded in the VideoObject instance. If no video is
        loaded, this method attempts to automatically load the video first by calling
        `load_video()`. If the video still cannot be loaded, a NoVideoLoadedError is raised.

        Raises:
            NoVideoLoadedError: If no video has been loaded into the VideoObject instance
                                and an automatic attempt to load the video fails. This
                                error indicates that the video could not be automatically
                                loaded, and manual loading may be required.
        """
        # Attempt to show the video if already loaded
        if self.video is not None:
            media.show_video(self.video, fps=self.video.metadata.fps)
        else:
            # Integrated fallback mechanism: attempt to load the video
            try:
                self.load_video()
                # Try showing the video again after loading
                if self.video is not None:
                    media.show_video(self.video, fps=self.video.metadata.fps)
                    self.release_video()  # if video was loaded automatically, release it afterwards
                else:
                    # Video still not loaded after attempt
                    raise NoVideoLoadedError("Failed to automatically load the video.")
            except Exception as e:
                # Handle specific exceptions related to loading failures
                raise NoVideoLoadedError("Video not loaded and automatic video loading failed. Error: " + str(e))

    def save_video_with_tracks(self, output_dir):
        video = self.get_video_with_tracks()
        media.write_video(os.path.join(output_dir, f"{self.video_id}.mp4"), video)

    def show_video_with_tracks(self):
        video = self.get_video_with_tracks()

        # Play the video
        media.show_video(video, fps=self.video.metadata.fps)

    def get_video_with_tracks(self):
        assert self.arranged_tracking_data is not {}, "No tracking data available for this video."

        # Ensure video is loaded
        self.load_video()

        num_points = len(self.arranged_tracking_data.keys())
        num_frames = len(self.video)

        height, width = self.video.shape[1:3]
        dot_size_as_fraction_of_min_edge = 0.015
        radius = int(round(min(height, width) * dot_size_as_fraction_of_min_edge))
        diam = radius * 2 + 1
        quadratic_y = np.square(np.arange(diam)[:, np.newaxis] - radius - 1)
        quadratic_x = np.square(np.arange(diam)[np.newaxis, :] - radius - 1)
        icon = (quadratic_y + quadratic_x) - (radius ** 2) / 2.0
        sharpness = 0.15
        icon = np.clip(icon / (radius * 2 * sharpness), 0, 1)
        icon = 1 - icon[:, :, np.newaxis]
        icon1 = np.pad(icon, [(0, 1), (0, 1), (0, 0)])
        icon2 = np.pad(icon, [(1, 0), (0, 1), (0, 0)])
        icon3 = np.pad(icon, [(0, 1), (1, 0), (0, 0)])
        icon4 = np.pad(icon, [(1, 0), (1, 0), (0, 0)])

        video = self.video.copy()
        for t in range(num_frames):
            # Pad so that points that extend outside the image frame don't crash us
            image = np.pad(
                video[t],
                [
                    (radius + 1, radius + 1),
                    (radius + 1, radius + 1),
                    (0, 0),
                ],
            )
            for i in self.arranged_tracking_data.keys():
                # The icon is centered at the center of a pixel, but the input coordinates
                # are raster coordinates.  Therefore, to render a point at (1,1) (which
                # lies on the corner between four pixels), we need 1/4 of the icon placed
                # centered on the 0'th row, 0'th column, etc.  We need to subtract
                # 0.5 to make the fractional position come out right.
                x = self.arranged_tracking_data[i][t]['x'] + 0.5
                y = self.arranged_tracking_data[i][t]['y'] + 0.5
                x = min(max(x, 0.0), width)
                y = min(max(y, 0.0), height)

                # if self.arranged_tracking_data[i][t]['visible']:
                if True:
                    x1, y1 = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
                    x2, y2 = x1 + 1, y1 + 1

                    # bilinear interpolation
                    patch = (
                            icon1 * (x2 - x) * (y2 - y)
                            + icon2 * (x2 - x) * (y - y1)
                            + icon3 * (x - x1) * (y2 - y)
                            + icon4 * (x - x1) * (y - y1)
                    )
                    x_ub = x1 + 2 * radius + 2
                    y_ub = y1 + 2 * radius + 2
                    image[y1:y_ub, x1:x_ub, :] = (1 - patch) * image[
                                                               y1:y_ub, x1:x_ub, :
                                                               ] + patch * np.array(self.colour_map[i])[np.newaxis,
                                                                           np.newaxis,
                                                                           :]

                # Remove the pad
                video[t] = image[
                           radius + 1: -radius - 1, radius + 1: -radius - 1
                           ].astype(np.uint8)

        return video

    def add_keypoint_labels(self, frame_index, labels):
        self.keypoint_labels[frame_index] = labels

    def get_keypoint_labels(self, frame_index, task):
        task_keypoints = {
            'extreme_keypoints': ['head top', 'left elbow', 'right elbow',
                                  'left wrist', 'right wrist', 'left knee',
                                  'right knee', 'left ankle', 'right ankle'],
            'all_body_keypoints': ['nose', 'head bottom', 'head top',
                                   'left ear', 'right ear', 'left shoulder',
                                   'right shoulder', 'left elbow', 'right elbow',
                                   'left wrist', 'right wrist', 'left hip',
                                   'right hip', 'left knee', 'right knee',
                                   'left ankle', 'right ankle']
        }

        # selected_keypoints = task_keypoints.get(task, [])
        frame_keypoints = self.keypoint_labels.get(frame_index, {})

        return {k: np.array([frame_keypoints[k][0], frame_keypoints[k][1]])
                for k in task_keypoints.get(task, []) if k in frame_keypoints}

    def label_and_store_keypoints(self, frame_index, task='extreme_keypoints'):
        # Assuming video_object is loaded TODO: raise error if video not loaded, try to load it as fallback mechanism
        video_frame = self.video[frame_index]

        point_labeler = PointLabeler()
        # point_labeler.label_points(video_frame, frame_index, task=task)
        point_labeler.label_points_adaptive(self, task)

        self.add_keypoint_labels(point_labeler.select_frame, point_labeler.selected_points)
        self.labeling_task = task

    def _get_filename(self, format, tag=None):
        if tag is None:
            return f"{self.video_id}.{self.labeling_task}.{format}"

        else:
            return f"{self.video_id}.{self.labeling_task}.{tag}.{format}"

    def _get_filename_with_tag_wildcard(self, format):
        # Return a string which can be used to glob files with the same video_id and task but different tags
        return f"{self.video_id}.{self.labeling_task}.*.{format}"

    def keypoints_file_exists(self, output_dir, task='extreme_keypoints', file_type='json'):
        file_types = ['csv', 'json']

        if file_type not in file_types:
            raise ValueError(f"Invalid file_type: {file_type}. Must be one of {file_types}")

        self.labeling_task = task
        file_path = os.path.join(output_dir, self._get_filename(file_type))
        return os.path.exists(file_path)

    def save_keypoints_to_csv(self, output_dir, tag=None):
        with open(os.path.join(output_dir, self._get_filename('csv', tag)), 'w', newline='') as csvfile:
            fieldnames = ['frame', 'keypoint', 'x_coord', 'y_coord']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for frame, points in self.keypoint_labels.items():
                for keypoint, coords in points.items():
                    # alternative: replace with placeholder value e.g. float('nan')
                    # x_coord, y_coord = (coords if coords is not None else (float('nan'), float('nan')))
                    if coords is not None:
                        writer.writerow({
                            'frame': frame,
                            'keypoint': keypoint,
                            'x_coord': coords[0],
                            'y_coord': coords[1]
                        })

    def save_keypoints_to_json(self, output_dir, tag=None, keypoints_to_save=None):
        """
        Saves keypoints to a JSON file. Can save adjusted (cropped) keypoints if provided.

        Parameters:
        - output_dir: Directory to save the JSON file.
        - tag: Optional tag to include in the filename.
        - adjusted_keypoints: Optional dictionary of adjusted keypoints. If not provided,
          the method uses the instance's keypoint_labels attribute.
        """
        keypoints_dict = keypoints_to_save if keypoints_to_save is not None else self.keypoint_labels

        # Ensure coordinates are integers before saving
        keypoints_dict = {
            frame: {
                keypoint: {'x': int(coords[0]), 'y': int(coords[1])}
                for keypoint, coords in frame_points.items() if coords is not None
            }
            for frame, frame_points in keypoints_dict.items()
        }

        # Write to JSON file
        filename = os.path.join(output_dir, self._get_filename('json', tag))
        with open(filename, 'w') as f:
            json.dump(keypoints_dict, f, indent=4)

    def load_outlier_data_from_folder(self, outlier_data_folder):
        file_path = os.path.join(outlier_data_folder, f'{self.video_id}_outliers.csv')

        # If this file does not exist, that only means there is no outlier data for this video
        if os.path.exists(file_path):
            self.outlier_data = pd.read_csv(file_path)

    def load_keypoint_labels_from_folder(self, labeled_keypoints_folder, task='extreme_keypoints', file_type='json'):
        file_types = ['csv', 'json']

        if file_type not in file_types:
            raise ValueError(f"Invalid file_type: {file_type}. Must be one of {file_types}")

        self.labeling_task = task

        file_path = os.path.join(labeled_keypoints_folder, self._get_filename(file_type))

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"No labeled keypoints file found for video ID '{self.video_id}' with task '{task}' and "
                f"file type '{file_type}'. {file_path} doesn't exist.")

        if file_type == 'json':
            self.keypoint_labels = self._load_keypoints_from_json(file_path)

        elif file_type == 'csv':
            self.keypoint_labels = self._load_keypoints_from_csv(file_path)

    def _load_keypoints_from_json(self, json_file_path):
        with open(json_file_path, 'r') as f:
            keypoint_labels = json.load(f)

        # frame index must be read as int, not string
        keypoint_labels = {int(frame_index): keypoints for frame_index, keypoints in keypoint_labels.items()}

        return keypoint_labels

    def _load_keypoints_from_csv(self, csv_file_path):
        keypoint_labels = {}
        with open(csv_file_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                frame_index = int(row['frame'])
                keypoint = row['keypoint']
                x_coord = float(row['x_coord'])
                y_coord = float(row['y_coord'])
                if frame_index not in keypoint_labels:
                    keypoint_labels[frame_index] = {}
                keypoint_labels[frame_index][keypoint] = np.array([x_coord, y_coord])

        return keypoint_labels

    def save_tracking_data(self, frame_index, tracks, visibles):
        self.tracking_data[frame_index] = (tracks, visibles)

    def load_labeled_keypoint_sets(self, labeled_keypoints_folder, task='extreme_keypoints', file_type='json'):
        # Glob all label sets for the video
        self.labeling_task = task
        matching_files = glob.glob(
            os.path.join(labeled_keypoints_folder, self._get_filename_with_tag_wildcard(file_type)))

        # Load the labeled keypoint sets from the folder and return them
        labeled_keypoint_sets = {}

        for file in matching_files:
            # Parse the tag out of the filename
            tag = os.path.basename(file).split('.')[2]

            if file_type == 'json':
                labeled_keypoint_sets[tag] = self._load_keypoints_from_json(file)
            elif file_type == 'csv':
                labeled_keypoint_sets[tag] = self._load_keypoints_from_csv(file)

        return labeled_keypoint_sets

    def merge_points(self, point_sets, labeled_frame_indices, task='extreme_keypoints', auto_accept_single=False):
        point_merger = PointMerger()
        for frame_index in labeled_frame_indices:
            point_merger.merge_points(self.video[frame_index], frame_index, point_sets, task, auto_accept_single)
            self.add_keypoint_labels(frame_index, point_merger.selected_points)

        self.labeling_task = task

    def track_points(self, tracker, task, labeled_keypoints_folder=None, file_type='json'):
        """
        Track points in the video. The method identifies and
        tracks keypoints defined by the task in the given frame
        and stores the tracking data.

        Args:
            tracker (PointTracker): The point tracking object.
            task (str): The task for which keypoints are to be tracked. Acceptable
                        values are 'extreme_keypoints' or 'all_body_keypoints'.
            labeled_keypoints_folder (str)

        Returns:
            Tuple of (tracks, visibles).
                - tracks: An array of tracked points across frames.
                - visibles: An array indicating the visibility of each point.

        Raises:
            ValueError: If the provided task is not among the acceptable options
        """
        possible_tasks = ['extreme_keypoints', 'all_body_keypoints']
        if task not in possible_tasks:
            raise ValueError(f"Invalid task: {task}. Must be one of {possible_tasks}")

        # Ensure keypoint labels are loaded for the frame_index
        if self.keypoint_labels == {}:
            if labeled_keypoints_folder:
                self.load_keypoint_labels_from_folder(labeled_keypoints_folder, task, file_type)
            else:
                raise ValueError(f"Keypoint labels are not already loaded, "
                                 f"please provide labelled keypoint folder.")

        # Build a dictionary of keypoints, each key having a list of frames where this keypoint is lableled
        keypoint_frames = defaultdict(lambda: np.zeros(len(self.video)))
        for frame_index, keypoints in self.keypoint_labels.items():
            for keypoint in keypoints.keys():
                keypoint_frames[keypoint][frame_index] = 1

        # Collect intervals to track, both forwards and backwards
        interval_dict = defaultdict(list)
        for keypoint, frames in keypoint_frames.items():
            # Identify the indices in frames where the keypoint is labeled
            labeled_indices = np.where(frames == 1)[0]
            num_labeled_indices = len(labeled_indices)

            for i, idx in enumerate(labeled_indices):
                # If this is the first labeled frame, and it is not the first frame, track backwards before tracking
                # forwards
                if i == 0 and idx > 0:
                    interval_dict[(0, idx, True)].append(keypoint)

                # If this is not the last labeled frame, track forwards until the next labeled frame
                if i < num_labeled_indices - 1:
                    interval_dict[(idx, labeled_indices[i + 1], False)].append(keypoint)

                # Else if this is the last labeled frame, track forwards until the end of the video
                elif i == num_labeled_indices - 1:
                    interval_dict[(idx, len(self.video), False)].append(keypoint)

        # Convert interval_dict to a list of frame intervals of the format: [(start_frame, end_frame, [labeled points], track_backward), ...]
        frame_intervals = []
        for interval, points in interval_dict.items():
            start_frame, end_frame, track_backward = interval
            frame_intervals.append((start_frame, end_frame, points, track_backward))

        for interval in frame_intervals:
            start_frame_index, end_frame_index, points, track_backward = interval

            if track_backward:
                keypoint_labels_frame = end_frame_index
            else:
                keypoint_labels_frame = start_frame_index

            labelled_keypoints = self.get_keypoint_labels(keypoint_labels_frame, task)
            labelled_keypoints = {k: v for k, v in labelled_keypoints.items() if k in points}  # Only keep the points in the interval

            # labelled_keypoints is a dictionary with keypoint names as key
            tracks, visibles = tracker.track_selected_points(self.video, start_frame_index, end_frame_index,
                                                             labelled_keypoints, track_backward)

            # Initialize an empty dictionary to store the results as a dictionary, where each value is a list of
            # dictionaries with coordinates x, y and whether the point is visible at this time frame
            # Format: {'keypoint': [{'x': x_coord, 'y': y_coord, 'visible': Array(True, dtype=bool)}, ...], ...}
            tracking_results = {}

            # Retrieve the task keypoints for the names
            task_keypoints = labelled_keypoints.keys()

            # Loop over each keypoint index and name
            for i, keypoint in enumerate(task_keypoints):
                tracking_results[keypoint] = []
                for frame in range(tracks.shape[1]):
                    coord = tracks[i, frame]
                    visible = visibles[i, frame]
                    tracking_results[keypoint].append({"x": coord[0], "y": coord[1], "visible": bool(visible)})

            self._merge_tracking_data(keypoint_labels_frame, tracking_results, track_backward)

    def _merge_tracking_data(self, frame_index, tracking_results, track_backward):
        if frame_index not in self.tracking_data.keys():
            self.tracking_data[frame_index] = tracking_results

        else:
            # Merge in the new tracking results
            for keypoint, data_list in tracking_results.items():
                if keypoint in self.tracking_data[frame_index]:
                    # If tracking backward, prepend the new data to the existing data
                    if track_backward:
                        self.tracking_data[frame_index][keypoint] = data_list + self.tracking_data[frame_index][keypoint]
                    else:
                        self.tracking_data[frame_index][keypoint] += data_list
                else:
                    self.tracking_data[frame_index][keypoint] = data_list

    def save_tracked_points_to_csv(self, folder):
        filename = os.path.join(folder, f'tracked_points_{self.video_id}.csv')
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write the header
            writer.writerow(['tracked_from_frame', 'keypoint', 'frame_index', 'x', 'y', 'visible'])

            # Iterate through the tracking data and write each point
            written_frame_counts = {}  # Track how many frames are written per keypoint

            for frame_index in sorted(self.tracking_data.keys()):
                tracking_result = self.tracking_data[frame_index]
                for keypoint, data_list in tracking_result.items():
                    for frame, data in enumerate(data_list):
                        writer.writerow([
                            frame_index,
                            keypoint,
                            written_frame_counts.get(keypoint, 0),
                            data['x'],
                            data['y'],
                            data['visible']
                        ]   )
                        written_frame_counts[keypoint] = written_frame_counts.get(keypoint, 0) + 1
        print('tracking_data saved to', filename)

    def save_tracked_points_to_json(self, folder):
        filename = os.path.join(folder, f'tracked_points_{self.video_id}.json')
        with open(filename, 'w') as f:
            # Use json.dump to write the data to a file, with indentation for readability
            json.dump(self.tracking_data, f, indent=4)
            print('tracking_data dumped to', filename)

    def load_tracked_points_from_folder(self, tracked_keypoints_folder, file_type='json'):
        file_types = {'csv': '.csv', 'json': '.json'}

        if file_type not in file_types:
            raise ValueError(f"Invalid file_type: {file_type}. Must be one of {list(file_types.keys())}")

        file_path = os.path.join(tracked_keypoints_folder, f'tracked_points_{self.video_id}{file_types[file_type]}')

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"No tracked_keypoints file found for video ID '{self.video_id}' with file type '{file_type}'.")

        if file_type == 'json':
            with open(file_path, 'r') as f:
                self.tracking_data = json.load(f)
            # frame index must be read as int, not string
            self.tracking_data = {int(frame_index): tracking_dict
                                  for frame_index, tracking_dict in self.tracking_data.items()}

        elif file_type == 'csv':
            self._load_tracked_points_from_csv(file_path)

        self.update_arranged_tracked_data()

    def update_arranged_tracked_data(self):
        self.arranged_tracking_data = self.get_arranged_tracking_data()

    def _load_tracked_points_from_csv(self, csv_file_path):

        # Initialize a nested dictionary
        tracking_results = defaultdict(lambda: defaultdict(list))

        tracking_data = {}
        with open(csv_file_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                # Convert string values to appropriate types
                frame_index = int(row['tracked_from_frame'])
                x = float(row['x'])
                y = float(row['y'])
                visible = row['visible'] == 'True'  # Convert string to boolean

                # Append the tracking point to the appropriate lists
                tracking_results[frame_index][row['keypoint']].append({
                    'x': x,
                    'y': y,
                    'visible': visible
                })

        self.tracking_data = {k: dict(v) for k, v in tracking_results.items()}

    def get_tracked_points(self):
        pass

    def update_extreme_coordinates(self, tracked_keypoints_folder=None, file_type='json'):
        """
        Update the extreme coordinates using tracked data_points.
        """
        if tracked_keypoints_folder:
            self.load_tracked_points_from_folder(tracked_keypoints_folder, file_type)
        elif self.tracking_data == {}:
            raise ValueError("No tracked points loaded, cannot update extreme coordinate. "
                             "Please provide tracked keypoints folder to load from.")

        self.extreme_coordinates = find_extreme_coordinates(self.tracking_data)

    # TODO: resize_height, resize_width as optional argument when resize=True or as fixed class parameter?
    def crop_and_resize_video(self, cropped_folder, resize=True, resize_folder=None, load_and_release_video=True):

        # Check extreme_coordinates available, try to update them if not try to use labeled points?
        if self.extreme_coordinates == {}:
            try:
                self.update_extreme_coordinates()
                print("Extreme coordinates have been successfully updated.")
            # TODO: try to use labelled_keypoints with a clear warning to get extreme_coordinates instead if available
            except ValueError as e:
                print(f"Warning: No extreme coordinates available for video {self.video_id} and no tracking data "
                      f"available to compute them.")
                return

        # video must be loaded to crop it and save it
        if load_and_release_video:
            self.load_video()
        elif self.video is None:
            raise ValueError(f'Video array for video {self.video_id} not loaded '
                             f'and load_and_release_video flag set to False')

        height, width = self.video.metadata.shape
        fps_value = self.video.metadata.fps
        bps_value = self.video.metadata.bps

        x_min = self.extreme_coordinates['leftmost']
        y_max = self.extreme_coordinates['topmost']
        x_max = self.extreme_coordinates['rightmost']
        y_min = self.extreme_coordinates['bottommost']

        # Get spread of selected points
        x_spread = x_max - x_min
        y_spread = y_max - y_min

        # Crop around selected points
        margin = 0.15  # 15% margin
        x_margin = round(margin * x_spread)
        y_margin = round(margin * y_spread)
        largest_margin = max(x_margin, y_margin)

        min_x_crop, max_x_crop = (max(0, x_min - largest_margin - 1),
                                  min(width, x_max + largest_margin + 1))
        min_y_crop, max_y_crop = (max(0, y_min - largest_margin - 1),
                                  min(height, y_max + largest_margin + 1))

        # -- Get the labelled keypoints in the cropped video coordinates
        cropped_keypoint_labels = {
            frame: {
                keypoint: {
                    'x': coords['x'] - min_x_crop,
                    'y': coords['y'] - min_y_crop
                }
                for keypoint, coords in frame_data.items()
            }
            for frame, frame_data in self.keypoint_labels.items()
        }

        # -- Save the keypoints in the cropped video coords to a file
        output_dir = './output/labeled'  # TODO: think about a more flexible way to do that
        tag = 'cropped'
        filename = os.path.join(output_dir, self._get_filename('json', tag))

        self.save_keypoints_to_json(
            output_dir=output_dir,
            tag=tag,
            keypoints_to_save=cropped_keypoint_labels
        )
        print(f'wrote labeled keypoints in cropped video coordinate space to {os.getcwd()}/{filename}')

        frames_original = self.video.__array__()
        frames_cropped = frames_original[:, min_y_crop:max_y_crop, min_x_crop:max_x_crop, :]

        if resize:
            resize_height = 256
            resize_width = 256
            # Expand to square (add background padding) and resize to tracker dimensions
            frames_crop_resized = []
            for frame in frames_cropped:
                im = Image.fromarray(frame, mode="RGB")
                squared_frame = expand2square(im, (255, 255, 255))
                resized_frame = squared_frame.resize((resize_height, resize_width))
                frames_crop_resized.append(np.asarray(resized_frame))

        # Save cropped video to file
        filename_cropped = f'cropped_vid_{self.video_id}.mp4'

        media.write_video(
            os.path.join(cropped_folder, filename_cropped),
            frames_cropped,
            fps=fps_value,
            bps=bps_value
        )
        print('wrote cropped video to', os.path.join(cropped_folder, filename_cropped))

        # Save cropped and resized video to file if argument is True
        if resize:
            filename_cropped_resized = f'cropped_resized_vid_{self.video_id}.mp4'
            media.write_video(
                os.path.join(resize_folder, filename_cropped_resized),
                frames_crop_resized,
                fps=fps_value,
                bps=bps_value
            )
            print('wrote cropped and resized video to',
                  os.path.join(resize_folder, filename_cropped_resized))

        # Release video from memory after processing it
        if load_and_release_video:
            self.release_video()

    def get_tracked_points_deltas(self):
        # Get video dimensions to make deltas relative
        if self.video is None:
            self.load_video()

        width, height = self.video.shape[2], self.video.shape[1]

        self.release_video()

        # Return the difference between successive frames for each tracked point
        frame_deltas = {keypoint: [] for keypoint in self.arranged_tracking_data}
        for keypoint, data_list in self.arranged_tracking_data.items():
            for i, data in enumerate(data_list[:-1]):
                x_diff = abs(data['x'] - data_list[i + 1]['x']) / width
                y_diff = abs(data['y'] - data_list[i + 1]['y']) / height
                frame_deltas[keypoint].append((x_diff, y_diff))

        return frame_deltas

    def get_arranged_tracking_data(self):
        assert self.tracking_data is not {}, f"No tracked points found"
        # arranged_tracking_data = {keypoint: [] for keypoint in self.tracking_data[list(self.tracking_data.keys())[0]]}
        arranged_tracking_data = {}

        for frame_index, data in self.tracking_data.items():
            for keypoint, data_list in data.items():
                if keypoint not in arranged_tracking_data:
                    arranged_tracking_data[keypoint] = []

                arranged_tracking_data[keypoint] += [
                    {'x': data_elem['x'], 'y': data_elem['y'], 'visible': data_elem['visible']} for data_elem
                    in data_list]

        return arranged_tracking_data

    def impute_value(self, keypoint, frame_index):
        assert frame_index != 0 and frame_index != len(self.video) - 1, "Cannot impute value for first or last frame"
        x_0, y_0 = self.arranged_tracking_data[keypoint][frame_index - 1]['x'], self.arranged_tracking_data[keypoint][frame_index - 1]['y']
        x_2, y_2 = self.arranged_tracking_data[keypoint][frame_index + 1]['x'], self.arranged_tracking_data[keypoint][frame_index + 1]['y']
        return np.mean([x_0, x_2]), np.mean([y_0, y_2])

    def get_tracking_data_position(self, frame_index):
        keyframes = sorted(list(self.tracking_data.keys()))
        keyframes.append(len(self.video))

        for i in range(len(keyframes) - 1):
            if keyframes[i] <= frame_index < keyframes[i + 1]:
                return keyframes[i], frame_index - keyframes[i]


    def get_sorted_outlier_table(self):
        return self.outlier_data.sort_values(by=['Keypoint', 'Outlier frame index'])

    def get_one_off_outliers(self, diff=1):
        outlier_table = self.get_sorted_outlier_table()
        outlier_table['diff'] = outlier_table['Outlier frame index'].diff(-1) * -1
        outlier_table['next_type'] = outlier_table['Outlier Type'].shift(-1)
        outlier_table['next_keypoint'] = outlier_table['Keypoint'].shift(-1)
        one_offs = outlier_table[(outlier_table['diff'] == diff) & (outlier_table['Outlier Type'] == 1) & (outlier_table['next_type'] == 3) & (outlier_table['Keypoint'] == outlier_table['next_keypoint'])]
        return one_offs

    def get_point_set_for_outlier(self, outlier_frame_index):
        # Load all points at the provided outlier frame index
        point_set = {}

        for kp, tracked_list in self.arranged_tracking_data.items():
            point_set[kp] = np.array([tracked_list[outlier_frame_index]['x'], tracked_list[outlier_frame_index]['y']])

        return point_set


#################################################
############## PatientData Class ################
#################################################

class PatientData:
    def __init__(self, age_group, health_status):
        self.age_group = age_group
        self.health_status = health_status


#################################################
######### NoVideoLoadedError Class ##############
#################################################

class NoVideoLoadedError(Exception):
    """Exception raised when attempting to show a video that hasn't been loaded."""
    pass
