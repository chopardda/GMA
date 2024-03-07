#     Purpose: Label specific points in the video frames.
#     Methods:
#         label_point(frame, point_id, label): Label a specific point in a frame.
#         get_labels(frame): Retrieve labels of points in a frame.
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from tapnet.utils import viz_utils


def create_bodypart_colormap(body_keypoints):
    # Use a matplotlib colormap
    colorpalette = sns.color_palette("hls", len(body_keypoints))  # 'tab20' is a good palette for distinct colors

    bodypart_colors = {
        body_keypoints[i]: colorpalette[i] for i in range(len(body_keypoints))
    }

    return bodypart_colors


class PointLabeler:
    def __init__(self):
        self.select_frame = None
        self.frame = None
        full_set_body_keypoints = ['nose',
                                   'head bottom', 'head top',
                                   'left ear', 'right ear',
                                   'left shoulder', 'right shoulder',
                                   'left elbow', 'right elbow',
                                   'left wrist', 'right wrist',
                                   'left hip', 'right hip',
                                   'left knee', 'right knee',
                                   'left ankle', 'right ankle']
        self.body_keypoints = full_set_body_keypoints  # default, defined in label_points
        self.max_n_points = len(full_set_body_keypoints)
        self.colormap = create_bodypart_colormap(full_set_body_keypoints)
        self.fig, self.ax_image, self.ax_list = None, None, None

        self.selected_points = {}
        # # Stores the positions of each body part for each frame
        # self.keypoint_positions = {}  # { frame_number: { keypoint: (x, y), ...}, ... }

    def setup_figure(self, frame):
        fig = plt.figure(figsize=(20, 10))
        ax_image = fig.add_subplot(121)
        ax_list = fig.add_subplot(122)
        ax_image.imshow(frame)  # self.video_data.video[self.select_frame])
        ax_image.axis('off')
        fig.suptitle('Click to select points. \n Spacebar to skip points. \n Afterwards, press enter to close '
                           'the figure.')
        ax_list.axis('off')
        self.update_list(ax_list)
        return fig, ax_image, ax_list

    def update_list(self, ax_list):
        ax_list.clear()
        ax_list.axis('off')
        for i, keypoint in enumerate(self.body_keypoints):
            text = keypoint + ': '
            color = self.colormap[keypoint]
            if keypoint in self.selected_points:
                text += str(self.selected_points[keypoint])
            ax_list.text(0, 1 - (i * 0.1), text, va='top', ha='left', fontsize=12, color=color)
        ax_list.figure.canvas.draw()

    def redraw_points(self):
        """
        Redraws the points on the image.
        """
        self.ax_image.clear()
        self.ax_image.imshow(self.frame) #self.video_data.video[self.select_frame])
        for keypoint in self.body_keypoints:
            if keypoint in self.selected_points:
                point = self.selected_points[keypoint]
                if point is not None:
                    color = self.colormap[keypoint]
                    self.ax_image.plot(point[0], point[1], 'o', color=color)
        self.ax_image.axis('off')
        self.update_list(self.ax_list)
        plt.draw()

    def on_click(self, event):
        if event.inaxes == self.ax_image:
            x, y = int(np.round(event.xdata)), int(np.round(event.ydata))
            current_keypoint = self.body_keypoints[len(self.selected_points)]
            self.selected_points[current_keypoint] = np.array([x, y])
            self.redraw_points()

    def on_key(self, event):
        if event.key == ' ':
            current_keypoint = self.body_keypoints[len(self.selected_points)]
            self.selected_points[current_keypoint] = None  # None for skipped points
            self.update_list(self.ax_list)
        elif event.key == 'backspace':
            if self.selected_points:
                last_keypoint = self.body_keypoints[len(self.selected_points) - 1]
                self.selected_points.pop(last_keypoint)  # Remove the last selected
        elif event.key == 'enter':
            plt.close(self.fig)
        self.redraw_points()

    def label_points(self, frame, frame_index, task='extreme_keypoints'):
        self.frame = frame
        self.select_frame = frame_index
        self.selected_points = {}  # reinitialise each time

        # -- Predefined body parts
        if task == 'extreme_keypoints':
            self.body_keypoints = ['head top',
                                   'left elbow', 'right elbow',
                                   'left wrist', 'right wrist',
                                   'left knee', 'right knee',
                                   'left ankle', 'right ankle']
        elif task == 'all_body_keypoints':
            self.body_keypoints = ['nose',
                                   'head bottom', 'head top',
                                   'left ear', 'right ear',
                                   'left shoulder', 'right shoulder',
                                   'left elbow', 'right elbow',
                                   'left wrist', 'right wrist',
                                   'left hip', 'right hip',
                                   'left knee', 'right knee',
                                   'left ankle', 'right ankle']
        else:
            raise ValueError(
                "task {} not recognized. Valid options are 'extreme_keypoints' and 'all_body_keypoints'.".format(task))

        self.max_n_points = len(self.body_keypoints)
        # self.colormap = viz_utils.get_colors(self.max_n_points)  # TODO: always same colormap

        self.fig, self.ax_image, self.ax_list = self.setup_figure(frame)
        self.redraw_points()

        cid_mouse = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        plt.show()

        # self.keypoint_positions[frame_index] = self.selected_points

    def get_labels(self, frame_index):
        return self.selected_points

    # def label_all_points(self, frame_number):
    #     pass
    #
    # def label_extreme_points(self, video, frame_number):
    #
    #     colormap = viz_utils.get_colors(self.max_n_points)
    #     selected_points = []
    #
    #     fig = plt.figure(figsize=(10, 5))
    #     ax_image = fig.add_subplot(121)  # Subplot for the image
    #     ax_list = fig.add_subplot(122)  # Subplot for the list of points
    #     ax_image.imshow(video[frame_number])  # Display the selected frame from the video
    #     ax_image.axis('off')
    #     ax_image.set_title(
    #         'Click to select points. \n Spacebar to skip points. \n Afterwards, press enter to close the figure.')
    #     ax_list.axis('off')
    #
    #     self.update_list(ax_list, self.max_n_points, selected_points, colormap)
    #
    #     cid_mouse = fig.canvas.mpl_connect('button_press_event',
    #                                        lambda event: self.on_click(event, selected_points))  # on_click)
    #     cid_key = fig.canvas.mpl_connect('key_press_event',
    #                                      lambda event: self.on_key(event, fig, ax_list, selected_points))
    #
    #     plt.show()
    #
    # def label_point(self, frame_number, body_part, position):
    #     """
    #     Labels a specific body part with its position in a frame.
    #
    #     :param frame_number: The frame number.
    #     :param body_part: The body part to label.
    #     :param position: The position (x, y) of the body part.
    #     """
    #     if body_part not in self.body_parts:
    #         raise ValueError(f"Invalid body part. Valid parts are: {self.body_parts}")
    #
    #     if frame_number not in self.positions:
    #         self.positions[frame_number] = {}
    #     self.positions[frame_number][body_part] = position
    #
    # def get_labels(self, frame_number):
    #     """
    #     Retrieves positions of body parts in a specified frame.
    #
    #     :param frame_number: The frame number.
    #     :return: A dictionary of body part positions for the frame, or None if no positions exist.
    #     """
    #     return self.positions.get(frame_number, None)
    #
    # def remove_label(self, frame_number, body_part):
    #     """
    #     Removes the label of a specific body part in a frame.
    #
    #     :param frame_number: The frame number.
    #     :param body_part: The body part to remove.
    #     """
    #     if frame_number in self.positions and body_part in self.positions[frame_number]:
    #         del self.positions[frame_number][body_part]
    #
    # # Function to update the list of points
    # def update_list(self, ax_list, body_parts, selected_points, colormap):
    #     ax_list.clear()
    #     ax_list.axis('off')
    #     for i, part in enumerate(body_parts):
    #         text = part + ': '
    #         color = 'black'  # Default color
    #         if i < len(selected_points):
    #             text += str(selected_points[i])
    #             color = tuple(np.array(colormap[i]) / 255.0)  # Set color for the text
    #         ax_list.text(0, 1 - (i * 0.1), text, va='top', ha='left', fontsize=12, color=color)
    #
    # # Event handler for mouse clicks
    # def on_click(event, select_points):
    #     if event.button == 1 and event.inaxes == ax_image:  # Left mouse button clicked
    #         x, y = int(np.round(event.xdata)), int(np.round(event.ydata))
    #
    #         # global select_points
    #         select_points.append(np.array([x, y]))
    #         update_list()
    #         # print(select_points)
    #
    #         color = colormap[len(select_points) - 1]
    #         color = tuple(np.array(color) / 255.0)
    #         ax_image.plot(x, y, 'o', color=color, markersize=5)
    #         plt.draw()
    #
    #         # if len(select_points) == 2:
    #         #     fig.canvas.mpl_disconnect(cid)
    #
    #         return select_points
    #
    # # Event handler for keyboard presses
    # def on_key(self, event, fig, ax_list):
    #     global selected_points
    #
    #     # -- Skip points that are not visible with spacebar
    #     if event.key == ' ':
    #         selected_points.append(np.array([None, None]))  # Add an empty point
    #         self.update_list(ax_list, self.extreme_parts, selected_points, colormap)
    #         plt.draw()
    #
    #         # print(select_points)
    #
    #     # -- Close figure with enter
    #     # TODO: cannot add more points once size of body_part list has been reached
    #     elif event.key == 'enter':
    #         plt.close(fig)
