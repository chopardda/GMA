import matplotlib
from matplotlib import pyplot as plt


class OutlierDisplay:
    def __init__(self, video_object):
        self.video_object = video_object
        self.fig, self.ax_left_image, self.ax_right_image = None, None, None

    def show_outlier(self, keypoint_name, frame_index, outlier_frame_index):
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
        self.ax_right_image.plot(right_point[0], right_point[1], 'o', color = 'red')

        # Plot arrow
        self.ax_right_image.arrow(left_point[0], left_point[1], right_point[0] - left_point[0], right_point[1] - left_point[1], width=1, color='blue', length_includes_head=True)

        self.fig.suptitle(f'Potential outlier for keypoint {keypoint_name} at frame {outlier_frame_index}')
        self._move_figure(200, 200)
        plt.show()
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
