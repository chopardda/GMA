import csv
import json
import math
import mediapy as media
import numpy as np
import os

from PIL import Image
from point_labeler import PointLabeler


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


def find_extreme_coordinates(tracks, visibles):
    # Initialize extreme coordinates with first values
    x_coords = [pt[0] for pt in tracks[:, 0, :]]
    y_coords = [pt[1] for pt in tracks[:, 0, :]]

    topmost, bottommost = max(y_coords), min(y_coords)  # get top- and bottom-most selected points
    rightmost, leftmost = max(x_coords), min(x_coords)  # get right- and left-most selected points

    num_keypoints, num_frames, _ = tracks.shape

    for keypoint in range(num_keypoints):
        for frame in range(num_frames):
            if visibles[keypoint, frame]:
                x, y = tracks[keypoint, frame]

                topmost = max(topmost, math.ceil(y))
                bottommost = min(bottommost, math.floor(y))
                leftmost = min(leftmost, math.floor(x))
                rightmost = max(rightmost, math.ceil(x))

    return leftmost, topmost, rightmost, bottommost


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


class VideoManager:
    def __init__(self):
        # Stores videos
        self.video_collection = {}  # Format: { video_id: VideoObject, ... }

    def add_video(self, filepath, load_now=False, add_pt_data=False):
        filename = os.path.basename(filepath)
        video_id = filename.split('_')[0]

        if load_now:
            video = media.read_video(filepath)
        else:
            video = None

        if add_pt_data:
            age_group, health_status = get_patient_data_from_filename(filename)
            patient_data = PatientData(age_group, health_status)
        else:
            patient_data = None

        self.video_collection[video_id] = VideoObject(filepath, video, patient_data)

    def add_all_videos(self, folder, load_now=False, add_pt_data=False):
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.lower().endswith('.mp4'):
                    self.add_video(os.path.join(root, file), load_now, add_pt_data)

    def get_all_video_ids(self):
        return list(self.video_collection.keys())

    def get_video_object(self, video_id):
        video_object = self.video_collection.get(video_id, None)
        return video_object if video_object else None

    def remove_video(self, video_id):
        if video_id in self.video_collection:
            del self.video_collection[video_id].video

    def show_video(self, video_id):
        vid = self.video_collection.get(video_id, None)
        media.show_video(vid.video, fps=vid.fps)

    def crop_and_resize_video(self, video_id, resize_height, resize_width):
        video = self.get_video_object(video_id).video
        height, width = video.metadata.shape
        fps_value = video.metadata.fps
        bps_value = video.metadata.bps

        x_min, y_max, x_max, y_min = video.get_extreme_coordinates(video_id)

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

        frames_original = video.__array__()
        frames_cropped = frames_original[:, min_y_crop:max_y_crop, min_x_crop:max_x_crop, :]

        # Expand to square (background padding) and resize to desired dimensions
        frames_crop_squared = []
        frames_crop_resized = []
        for frame in frames_cropped:
            im = Image.fromarray(frame, mode="RGB")
            squared_frame = expand2square(im, (255, 255, 255))
            resized_frame = squared_frame.resize((resize_height, resize_width))
            frames_crop_squared.append(np.asarray(squared_frame))
            frames_crop_resized.append(np.asarray(resized_frame))

        media.write_video(
            os.path.join('output', 'cropped_videos', f'cropped_vid_{video_id}.mp4'),
            frames_cropped,
            fps=fps_value,
            bps=bps_value
        )
        print('wrote cropped video to', os.path.join('output', 'cropped_videos', f'cropped_vid_{video_id}.mp4'))

        media.write_video(
            os.path.join('output', 'cropped_videos', f'cropped_resized_vid_{video_id}.mp4'),
            frames_crop_resized,
            fps=fps_value,
            bps=bps_value
        )
        print('wrote cropped and resized video to',
              os.path.join('output', 'cropped_videos', f'cropped_resized_vid_{video_id}.mp4'))


class VideoObject:
    def __init__(self, file_path, video=None, patient_data=None):
        self.file_path = file_path
        self.video = video  # a VideoArray object

        self.patient_data = patient_data

        self.keypoint_labels = {}  # Stores point labels for each video {frame_index: labels, ...}
        self.tracking_data = {}  # Stores tracking data {frame_index: (tracks, visibles)} #TODO: change
        self.extreme_coordinates = {}  # Format: {leftmost: , topmost: , rightmost: , bottommost: }

    def load_video(self):
        """
        Loads the video in the VideoObject if there is no video yet.
        """
        if not self.video:
            self.video = media.read_video(self.file_path)

    def release_video(self):
        if self.video is not None:
            self.video = None

    def add_keypoint_labels(self, frame_index, labels):
        self.keypoint_labels[frame_index] = labels

    def get_keypoint_labels(self, frame_index):
        return self.keypoint_labels[frame_index]

    def label_and_store_keypoints(self, frame_index):
        # Assuming video_object is loaded and accessible
        video_frame = self.video[frame_index]

        point_labeler = PointLabeler()
        point_labeler.label_points(video_frame, frame_index, task='extreme_keypoints')

        self.add_keypoint_labels(frame_index, point_labeler.selected_points)

    def save_keypoints_to_csv(self, filename):
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['frame', 'keypoint', 'x_coord', 'y_coord']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for frame, points in self.keypoint_labels.items():
                for keypoint, coords in points.items():
                    writer.writerow({
                        'frame': frame,
                        'keypoint': keypoint,
                        'x_coord': coords[0],
                        'y_coord': coords[1]
                    })

    def save_keypoints_to_json(self, filename):
        # Convert NumPy arrays to lists
        for frame, points in self.keypoint_labels.items():
            for keypoint, coords in points.items():
                self.keypoint_labels[frame][keypoint] = {'x': int(coords[0]), 'y': int(coords[1])}

        # Write to JSON file
        with open(filename, 'w') as f:
            json.dump(self.keypoint_labels, f, indent=4)
    def save_tracking_data(self, video_id, frame_index, tracks, visibles):
        if video_id not in self.tracking_data:
            self.tracking_data[video_id] = {}
        self.tracking_data[video_id][frame_index] = (tracks, visibles)

    def get_tracking_data(self, video_id, frame_index):
        return self.tracking_data.get(video_id, {}).get(frame_index, (None, None))

    def update_extreme_coordinates(self, video_id, tracker, video, frame_index, point_label):
        tracks, visibles = tracker.track_selected_points(video, frame_index, point_label)
        self.save_tracking_data(video_id, frame_index, tracks, visibles)
        extreme_coords = find_extreme_coordinates(tracks, visibles)
        self.extreme_coordinates[video_id] = extreme_coords

    def get_extreme_coordinates(self, video_id):
        return self.extreme_coordinates.get(video_id, None)


class PatientData:
    def __init__(self, age_group, health_status):
        self.age_group = age_group
        self.health_status = health_status
