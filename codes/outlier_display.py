import os

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import rc


class OutlierDisplay:
    def __init__(self, video_object, save_figures=False, stddev_threshold=3.0):
        self.video_object = video_object
        self.save_figures = save_figures
        self.stddev_threshold = stddev_threshold
        self.frame_index = None
        self.current_outlier_frame_index = None
        self.current_keypoint_name = None
        self.fig, self.ax_left_image, self.ax_right_image = None, None, None
        self.confirmed_outliers = {}

        # Ensure output directories exist
        if not os.path.exists('./output/outliers'):
            os.makedirs('./output/outliers', exist_ok=True)

        if save_figures and not os.path.exists('./output/outliers/figures'):
            os.makedirs('./output/outliers/figures', exist_ok=True)

    def show_outlier(self, keypoint_name, frame_index, outlier_frame_index, x_diff, x_stddev_mul, y_diff, y_stddev_mul):
        self.frame_index = frame_index
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
        left_point = self.video_object.tracking_data[0][keypoint_name][outlier_frame_index]['x'], \
            self.video_object.tracking_data[0][keypoint_name][outlier_frame_index]['y']

        right_point = self.video_object.tracking_data[0][keypoint_name][outlier_frame_index + 1]['x'], \
            self.video_object.tracking_data[0][keypoint_name][outlier_frame_index + 1]['y']
        self.ax_left_image.plot(left_point[0], left_point[1], 'o', color='red')
        self.ax_right_image.plot(right_point[0], right_point[1], 'o', color='red')

        # Plot arrow
        self.ax_right_image.arrow(left_point[0], left_point[1], right_point[0] - left_point[0],
                                  right_point[1] - left_point[1], width=1, color='blue', length_includes_head=True)

        self.fig.suptitle(f'Potential outlier for keypoint {keypoint_name} at frame {outlier_frame_index}\n'
                          f'X diff: {x_diff}, X stddev mul: {x_stddev_mul}\n'
                          f'Y diff: {y_diff}, Y stddev mul: {y_stddev_mul}\n\n'
                          f'Press Enter to confirm as an outlier, Escape to close the window.')

        self._move_figure(200, 200)
        cid_key = self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        plt.show()

    def write_outliers_to_file(self):
        with open(f'./output/outliers/{self.video_object.video_id}_outliers.csv', 'w') as f:
            f.write('Frame index, Keypoint, Outlier frame index\n')
            for frame_index, keypoint_outliers in self.confirmed_outliers.items():
                for keypoint, outlier_frame_indices in keypoint_outliers.items():
                    for outlier_frame_index in outlier_frame_indices:
                        f.write(f'{frame_index}, {keypoint}, {outlier_frame_index}\n')

    def _on_key(self, event):
        if event.key == 'escape':
            plt.close(self.fig)

        elif event.key == 'enter':
            if self.frame_index not in self.confirmed_outliers:
                self.confirmed_outliers[self.frame_index] = {}

            if self.current_keypoint_name not in self.confirmed_outliers[self.frame_index]:
                self.confirmed_outliers[self.frame_index][self.current_keypoint_name] = []

            self.confirmed_outliers[self.frame_index][self.current_keypoint_name].append(
                self.current_outlier_frame_index)

            if self.save_figures:
                self.fig.suptitle("")
                plt.savefig(f'./output/outliers/{self.video_object.video_id}_{self.frame_index}_{self.current_keypoint_name}_'
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
