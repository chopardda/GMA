import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from point_labeler import PointLabeler


class PointRelabeler(PointLabeler):
    def __init__(self):
        super().__init__()
        self.current_point = None

    def setup_figure(self, frame):
        fig = plt.figure(figsize=(20, 10))
        ax_image = fig.add_subplot(121)
        ax_list = fig.add_subplot(122)
        ax_image.imshow(frame)  # self.video_data.video[self.select_frame])
        ax_image.axis('off')
        fig.suptitle('Use the number keys to choose a point to redefine, and the click to choose the new location. \n'
                     'Afterwards, press enter to close the figure.')
        ax_list.axis('off')
        self.update_list(ax_list)
        return fig, ax_image, ax_list

    def update_list(self, ax_list):
        ax_list.clear()
        ax_list.axis('off')
        for i, keypoint in enumerate(self.body_keypoints):
            text = f'{i}. keypoint: '
            color = self.colormap[keypoint]
            if keypoint in self.selected_points:
                text += str(self.selected_points[keypoint])
            ax_list.text(0, 1 - (i * 0.1), text, va='top', ha='left', fontsize=12, color=color)
        ax_list.figure.canvas.draw()

    def on_click(self, event):
        if self.current_point is not None:
            if event.inaxes == self.ax_image:
                x, y = int(np.round(event.xdata)), int(np.round(event.ydata))
                self.selected_points[self.current_point] = np.array([x, y])
                self.redraw_points()
                self.current_point = None

    def on_key(self, event):
        if self.current_point is None:
            if event.key in ['0', '1', '2', '3', '4', '5', '6', '7', '8']:
                self.current_point = self.body_keypoints[int(event.key)]

            elif event.key == 'enter':
                plt.close(self.fig)

            elif event.key == 'escape':
                self.current_point = None

    def relabel_points(self, frame, frame_index, current_points, task='extreme_keypoints'):
        self.frame = frame
        self.select_frame = frame_index

        # Copy current_points into selected_points
        self.selected_points = current_points.copy()

        # -- Predefined body parts
        self._set_body_keypoints(task)
        self.max_n_points = len(self.body_keypoints)

        self.fig, self.ax_image, self.ax_list = self.setup_figure(frame)
        self.redraw_points()

        cid_mouse = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        plt.show()
