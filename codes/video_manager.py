import csv
import json
import math
import mediapy as media
import numpy as np
import os
import glob

from collections import defaultdict
from PIL import Image
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

    def add_video(self, filepath, load_now=False, add_pt_data=False):
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

        self.video_collection[video_id] = VideoObject(filepath, video_id, video, patient_data)

    def add_all_videos(self, folder, load_now=False, add_pt_data=False):
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.lower().endswith('.mp4'):
                    self.add_video(os.path.join(root, file), load_now, add_pt_data)

    def get_all_video_ids(self):
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

    def load_all_tracked_points(self, tracked_keypoints_folder, file_type='json'):
        for video_id, video_object in self.video_collection.items():
            video_object.load_tracked_points_from_folder(tracked_keypoints_folder, file_type)

#################################################
############# VideoObject Class ################
#################################################

class VideoObject:
    def __init__(self, file_path, video_id, video=None, patient_data=None):
        self.file_path = file_path
        self.video_id = video_id
        self.video = video  # a VideoArray object

        self.patient_data = patient_data

        self.keypoint_labels = {}  # Stores point labels for each video {frame_index: labels, ...}
        self.tracking_data = {}  # Stores tracked points {'point': [{'x': x_val, 'y':y_val, 'visible': bool), ...], ...}
        self.extreme_coordinates = {}  # Format: {leftmost: , topmost: , rightmost: , bottommost: }

        self.labeling_task = None

    def load_video(self):
        """
        Loads the video in the VideoObject if there is no video loaded yet.
        """
        if self.video is None:
            print(f'Loading video {self.video_id}')
            self.video = media.read_video(self.file_path)

    def release_video(self):
        """
        Release the video in the VideoObject if loaded.
        """
        if self.video is not None:
            print(f'Releasing video {self.video_id}')
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

        return {k: np.array([frame_keypoints[k]['x'], frame_keypoints[k]['y']])
                for k in task_keypoints.get(task, []) if k in frame_keypoints}

    def label_and_store_keypoints(self, frame_index, task='extreme_keypoints'):
        # Assuming video_object is loaded TODO: raise error if video not loaded, try to load it as fallback mechanism
        video_frame = self.video[frame_index]

        point_labeler = PointLabeler()
        point_labeler.label_points(video_frame, frame_index, task=task)

        self.add_keypoint_labels(frame_index, point_labeler.selected_points)
        self.labeling_task = task

    def _get_filename(self, format, tag=None):
        if tag is None:
            return f"{self.video_id}.{self.labeling_task}.{format}"

        else:
            return f"{self.video_id}.{self.labeling_task}.{tag}.{format}"

    def _get_filename_with_tag_wildcard(self, format):
        # Return a string which can be used to glob files with the same video_id and task but different tags
        return f"{self.video_id}.{self.labeling_task}.*.{format}"

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

    def save_keypoints_to_json(self, output_dir, tag=None):
        # Convert NumPy arrays to lists
        for frame, points in self.keypoint_labels.items():
            for keypoint, coords in points.items():
                if coords is not None:
                    self.keypoint_labels[frame][keypoint] = {'x': int(coords[0]), 'y': int(coords[1])}

        # Write to JSON file
        with open(os.path.join(output_dir, self._get_filename('json', tag)), 'w') as f:
            json.dump(self.keypoint_labels, f, indent=4)

    def load_keypoint_labels_from_folder(self, labeled_keypoints_folder, task='extreme_keypoints', file_type='json'):
        file_types = ['csv', 'json']

        if file_type not in file_types:
            raise ValueError(f"Invalid file_type: {file_type}. Must be one of {file_types}")

        self.labeling_task = task

        file_path = os.path.join(labeled_keypoints_folder, self._get_filename(file_type))

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"No labeled keypoints file found for video ID '{self.video_id}' with task '{task}' and "
                f"file type '{file_type}'.")

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

    def merge_points(self, point_sets, frame_index, task='extreme_keypoints', auto_accept_single=False):
        point_merger = PointMerger()
        point_merger.merge_points(self.video[frame_index], frame_index, point_sets, task, auto_accept_single)
        self.add_keypoint_labels(frame_index, point_merger.selected_points)
        self.labeling_task = task

    def track_points(self, tracker, frame_index, task, labeled_keypoints_folder=None, file_type='json'):
        """
        Track points in the video. The method identifies and
        tracks keypoints defined by the task in the given frame
        and stores the tracking data.

        Args:
            tracker (PointTracker): The point tracking object.
            frame_index (int): Index of the frame to start tracking.
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
        if frame_index not in self.keypoint_labels:
            if labeled_keypoints_folder:
                self.load_keypoint_labels_from_folder(labeled_keypoints_folder, task, file_type)
            else:
                raise ValueError(f"Keypoint labels for frame {frame_index} are not already loaded, "
                                 f"please provide labelled keypoint folder.")

        # Check if keypoint labels were successfully loaded
        if frame_index not in self.keypoint_labels:
            raise ValueError(f"Keypoint labels for frame {frame_index} could not be loaded.")

        labelled_keypoints = self.get_keypoint_labels(frame_index, task)

        # labelled_keypoints is a dictionary with keypoint names as key
        tracks, visibles = tracker.track_selected_points(self.video, frame_index, labelled_keypoints)

        # Initialize an empty dictionary to store the results as a dictionary, where each value is a list of
        # dictionaries with coordinates x, y and whether the point is visible at this time frame
        # Format: {'keypoint': [{'x': x_coord, 'y': y_coord, 'visible': Array(True, dtype=bool)}, ...], ...}
        tracking_results = {}

        # Retrieve the task keypoints for the names
        task_keypoints = self.get_keypoint_labels(frame_index, task).keys()

        # Loop over each keypoint index and name
        for i, keypoint in enumerate(task_keypoints):
            tracking_results[keypoint] = []
            for frame in range(tracks.shape[1]):
                coord = tracks[i, frame]
                visible = visibles[i, frame]
                tracking_results[keypoint].append({"x": coord[0], "y": coord[1], "visible": bool(visible)})

        self.tracking_data[frame_index] = tracking_results

    def save_tracked_points_to_csv(self, folder):
        filename = os.path.join(folder, f'tracked_points_{self.video_id}.csv')
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write the header
            writer.writerow(['tracked_from_frame', 'keypoint', 'frame_index', 'x', 'y', 'visible'])

            # Iterate through the tracking data and write each point
            for frame_index, tracking_result in self.tracking_data.items():
                for keypoint, data_list in tracking_result.items():
                    for frame, data in enumerate(data_list):
                        writer.writerow([
                            frame_index,
                            keypoint,
                            frame,
                            data['x'],
                            data['y'],
                            data['visible']
                        ])
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

        cropped_width = max_x_crop - min_x_crop
        cropped_height = max_y_crop - min_y_crop

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

        filename_cropped = f'cropped_vid_{self.video_id}.mp4'

        media.write_video(
            os.path.join(cropped_folder, filename_cropped),
            frames_cropped,
            fps=fps_value,
            bps=bps_value
        )
        print('wrote cropped video to', os.path.join(cropped_folder, filename_cropped))

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

        # release video from memory after processing it
        if load_and_release_video:
            self.release_video()

    def get_tracked_points_deltas(self, frame_index):
        # Return the difference between successive frames for each tracked point based on the frame_index
        assert frame_index in self.tracking_data, f"No tracked points found for frame {frame_index}"

        frame_deltas = { keypoint: [] for keypoint in self.tracking_data[frame_index] }
        for keypoint, data_list in self.tracking_data[frame_index].items():
            for i, data in enumerate(data_list[:-1]):
                x_diff = abs(data['x'] - data_list[i+1]['x'])
                y_diff = abs(data['y'] - data_list[i+1]['y'])
                frame_deltas[keypoint].append((x_diff, y_diff))

        return frame_deltas


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
