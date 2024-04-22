import os
from enum import Enum

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import rc
import pandas as pd


class OutlierDisplay:
    OUTLIER_DIR = './output/outliers'
    FIGURES_DIR = './output/outliers/figures'
    outlier_type = Enum('outlier_type', 'NOT_OUTLIER INITIAL ALREADY RETURN', start=0)

    def __init__(self, video_object, save_figures=False, stddev_threshold=3.0):
        self.video_object = video_object
        self.save_figures = save_figures
        self.stddev_threshold = stddev_threshold
        self.current_outlier_frame_index = None
        self.current_keypoint_name = None
        self.fig, self.ax_left_image, self.ax_right_image = None, None, None
        self.new_confirmed_outliers = []

        # Ensure output directories exist

        if not os.path.exists(OutlierDisplay.OUTLIER_DIR):
            os.makedirs(OutlierDisplay.OUTLIER_DIR, exist_ok=True)

        if save_figures and not os.path.exists(OutlierDisplay.FIGURES_DIR):
            os.makedirs(OutlierDisplay.FIGURES_DIR, exist_ok=True)

        # Load/create outlier table
        self.outlier_table_file = f'{OutlierDisplay.OUTLIER_DIR}/{self.video_object.video_id}_outliers.csv'

        if os.path.exists(self.outlier_table_file):
            self.confirmed_outliers_table = pd.read_csv(self.outlier_table_file)

        else:
            self.confirmed_outliers_table = pd.DataFrame(
                columns=['Keypoint', 'Outlier frame index', 'Outlier Type'])

    def show_outlier(self, keypoint_name, outlier_frame_index, x_diff, x_stddev_mul, y_diff, y_stddev_mul):
        # Check if outlier has already been confirmed
        if self.confirmed_outliers_table[
            (self.confirmed_outliers_table['Keypoint'] == keypoint_name) &
            (self.confirmed_outliers_table['Outlier frame index'] == outlier_frame_index)].shape[0] > 0:
            return

        # Load video if it is not already loaded
        if self.video_object.video is None:
            self.video_object.load_video()

        self.current_outlier_frame_index = outlier_frame_index
        self.current_keypoint_name = keypoint_name
        self.fig = plt.figure(figsize=(20, 10))
        self.ax_left_image = self.fig.add_subplot(121)
        self.ax_left_image.imshow(self.video_object.video[outlier_frame_index])
        self.ax_left_image.axis('off')
        self.ax_right_image = self.fig.add_subplot(122)
        self.ax_right_image.imshow(self.video_object.video[outlier_frame_index + 1])
        self.ax_right_image.axis('off')

        # Plot points
        left_point = self.video_object.arranged_tracking_data[keypoint_name][outlier_frame_index]['x'], \
            self.video_object.arranged_tracking_data[keypoint_name][outlier_frame_index]['y']

        right_point = self.video_object.arranged_tracking_data[keypoint_name][outlier_frame_index + 1]['x'], \
            self.video_object.arranged_tracking_data[keypoint_name][outlier_frame_index + 1]['y']
        self.ax_left_image.plot(left_point[0], left_point[1], 'o', color='red')
        self.ax_right_image.plot(right_point[0], right_point[1], 'o', color='red')

        # Plot arrow
        self.ax_right_image.arrow(left_point[0], left_point[1], right_point[0] - left_point[0],
                                  right_point[1] - left_point[1], width=1, color='blue', length_includes_head=True)

        self.fig.suptitle(f'Potential outlier for keypoint {keypoint_name} at frame {outlier_frame_index}\n'
                          f'X diff: {x_diff}, X stddev mul: {x_stddev_mul}\n'
                          f'Y diff: {y_diff}, Y stddev mul: {y_stddev_mul}\n\n'
                          f'Press Enter to confirm as an outlier, 1 to indicate as initial outlier, 2 to indicate as '
                          f'already an outlier and 3 to indicate as returning to place')

        self._move_figure(200, 200)
        cid_key = self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        plt.show()

    def write_outliers_to_file(self):
        # Append new confirmed outliers to outlier table and save it
        self.confirmed_outliers_table = pd.concat(
            [self.confirmed_outliers_table,
             pd.DataFrame(self.new_confirmed_outliers,
                          columns=['Keypoint', 'Outlier frame index', 'Outlier Type'])],
            ignore_index=True)

        # Save outliers table to file
        self.confirmed_outliers_table.to_csv(self.outlier_table_file, index=False)

    def _on_key(self, event):
        if event.key in ['escape', '1', '2', '3']:
            if self.confirmed_outliers_table[
                (self.confirmed_outliers_table['Keypoint'] == self.current_keypoint_name) &
                (self.confirmed_outliers_table['Outlier frame index'] == self.current_outlier_frame_index)].shape[
                0] == 0:
                # Determine type
                if event.key == 'escape':
                    outlier_type = OutlierDisplay.outlier_type.NOT_OUTLIER.value
                elif event.key == '1':
                    outlier_type = OutlierDisplay.outlier_type.INITIAL.value
                elif event.key == '2':
                    outlier_type = OutlierDisplay.outlier_type.ALREADY.value
                else:
                    outlier_type = OutlierDisplay.outlier_type.RETURN.value

                # Add outlier to new outliers list
                self.new_confirmed_outliers.append(
                    [self.current_keypoint_name, self.current_outlier_frame_index, outlier_type])

            if self.save_figures:
                self.fig.suptitle("")
                plt.savefig(
                    f'{OutlierDisplay.FIGURES_DIR}/{self.video_object.video_id}_{self.current_keypoint_name}_'
                    f'{self.current_outlier_frame_index}.png'.replace(' ', '_'))

            plt.close(self.fig)

    def _move_figure(self, x, y):
        """Move figure's upper left corner to pixel (x, y)"""
        backend = matplotlib.get_backend()
        if backend == 'TkAgg':
            self.fig.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
        elif backend == 'WXAgg':
            self.fig.canvas.manager.window.SetPosition((x, y))
        else:
            # This works for QT and GTK
            # You can also use window.setGeometry
            self.fig.canvas.manager.window.move(x, y)
