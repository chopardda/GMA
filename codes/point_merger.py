import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from point_labeler import PointLabeler



class PointMerger(PointLabeler):
    def __init__(self):
        super().__init__()
        self.selected_point_sets = {}

    def setup_figure(self, frame):
        fig = plt.figure(figsize=(20, 10))
        ax_image = fig.add_subplot(121)
        ax_list = fig.add_subplot(122)
        ax_image.imshow(frame)  # self.video_data.video[self.select_frame])
        ax_image.axis('off')
        fig.suptitle('Use the spacebar to accept the proposed merged point (in red). Click to define a new '
                           'final point. \n Afterwards, press enter to close the figure.')
        ax_list.axis('off')
        self.update_list(ax_list)
        return fig, ax_image, ax_list

    def redraw_points(self):
        """
        Redraws the points on the image.
        """
        self.ax_image.clear()
        self.ax_image.imshow(self.frame) #self.video_data.video[self.select_frame])

        # Draw all already selected points
        for keypoint in self.body_keypoints:
            if keypoint in self.selected_points:
                point = self.selected_points[keypoint]
                if point is not None:
                    self.ax_image.plot(point[0], point[1], 'o', color='red')

        # Draw all labeled points for the next unmerged body keypoint, as well as the average point (in red), provided
        # this is not the last redraw
        if len(self.selected_points) != len(self.body_keypoints):
            labeled_points, average_point = self._get_current_labeled_and_average_points()

            # Get a colour palette with no red
            color_palette = sns.color_palette("husl", len(labeled_points))
            self.ax_image.plot(average_point[0], average_point[1], 'o', color='red')

            # Plot the rest of the points
            for i, point in enumerate(labeled_points):
                self.ax_image.plot(point['x'], point['y'], 'o', color=color_palette[i])

            self.ax_image.axis('off')
            self.update_list(self.ax_list)

        plt.draw()

    def on_key(self, event):
        if event.key == ' ':
            current_keypoint = self.body_keypoints[len(self.selected_points)]
            _, average_point = self._get_current_labeled_and_average_points()
            self.selected_points[current_keypoint] = np.array(average_point)  # None for skipped points
            self.update_list(self.ax_list)
        elif event.key == 'backspace':
            if self.selected_points:
                last_keypoint = self.body_keypoints[len(self.selected_points) - 1]
                self.selected_points.pop(last_keypoint)  # Remove the last selected
        elif event.key == 'enter':
            plt.close(self.fig)
        self.redraw_points()

    def _get_current_labeled_and_average_points(self):
        labeled_points = []
        current_keypoint = self.body_keypoints[len(self.selected_points)]
        for k, v in self.selected_point_sets.items():
            labeled_points.append(v[self.select_frame][current_keypoint])

        assert len(labeled_points) > 0, "No labeled points found for the current keypoint."

        # Calculate the average point
        average_point = [sum([p['x'] for p in labeled_points]) / len(labeled_points),
                         sum([p['y'] for p in labeled_points]) / len(labeled_points)]

        return labeled_points, average_point

    def merge_points(self, frame, frame_index, point_sets, task='extreme_keypoints'):
        self.selected_point_sets = point_sets
        self.label_points(frame, frame_index, task)