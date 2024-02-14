import os

import matplotlib.pyplot as plt
import mediapy as media
import numpy as np

from tapnet.utils import viz_utils
from video_processing_helpers import expand2square

body_parts = ['0. nose',
              '1. head bottom',
              'right shoulder',
              'left shoulder',
              'right elbow', 'left elbow', 'right wrist', 'left wrist']
MAX_N_POINTS = len(body_parts)

# --- Load video
video = media.read_video('/home/daphne/Documents/GMA/data/Preprocessed_Videos/AI_GeMo_late_F-/35_F-_2_c.mp4')
height, width = video.shape[1:3]
media.show_video(video, fps=30)

select_frame = 0  # @param {type:"slider", min:0, max:49, step:1}

# Generate a colormap with MAX_N_POINTS points
np.random.seed(42)  # to ensure colormap is the same each time TODO: change to predefined map
colormap = viz_utils.get_colors(MAX_N_POINTS)

select_points = []

# --- Open first frame to select points
# --- TODO: function. Then also open last frame to later merge predictions
fig = plt.figure(figsize=(10, 5))
ax_image = fig.add_subplot(121)  # Subplot for the image
ax_list = fig.add_subplot(122)  # Subplot for the list of points
ax_image.imshow(video[select_frame])  # Display the selected frame from the video
ax_image.axis('off')
ax_image.set_title(
    'Click to select points. \n Spacebar to skip points. \n Afterwards, press enter to close the figure.')
ax_list.axis('off')


# Function to update the list of points
def update_list():
    ax_list.clear()
    ax_list.axis('off')
    for i, part in enumerate(body_parts):
        text = part + ': '
        color = 'black'  # Default color
        if i < len(select_points):
            text += str(select_points[i])
            color = tuple(np.array(colormap[i]) / 255.0)  # Set color for the text
        ax_list.text(0, 1 - (i * 0.1), text, va='top', ha='left', fontsize=12, color=color)


update_list()


# Event handler for mouse clicks
def on_click(event, select_points):
    if event.button == 1 and event.inaxes == ax_image:  # Left mouse button clicked
        x, y = int(np.round(event.xdata)), int(np.round(event.ydata))

        # global select_points
        select_points.append(np.array([x, y]))
        update_list()
        # print(select_points)

        color = colormap[len(select_points) - 1]
        color = tuple(np.array(color) / 255.0)
        ax_image.plot(x, y, 'o', color=color, markersize=5)
        plt.draw()

        # if len(select_points) == 2:
        #     fig.canvas.mpl_disconnect(cid)

        return select_points


# Event handler for keyboard presses
def on_key(event):
    global select_points

    # -- Skip points that are not visible with spacebar
    if event.key == ' ':
        select_points.append(np.array([None, None]))  # Add an empty point
        update_list()
        plt.draw()

        # print(select_points)

    # -- Close figure with enter
    # TODO: cannot add more points once size of body_part list has been reached
    elif event.key == 'enter':
        plt.close(fig)


cid_mouse = fig.canvas.mpl_connect('button_press_event', lambda event: on_click(event, select_points))  # on_click)
cid_key = fig.canvas.mpl_connect('key_press_event', on_key)

plt.show()

print('>>>>>>>>>>>')
print(select_points)
print('<<<<<<<<<<')
