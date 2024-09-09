import matplotlib.pyplot as plt
import mediapy as media
import numpy as np
import seaborn as sns


def create_bodypart_colormap(body_keypoints):
    # Use a matplotlib colormap
    colorpalette = sns.color_palette("hls", len(body_keypoints))  # 'tab20' is a good palette for distinct colors
    bodykeypoints_colors = {
        body_keypoints[i]: colorpalette[i] for i in range(len(body_keypoints))
    }

    return bodykeypoints_colors


body_keypoints = ['0. head top',
                  '1. left elbow',
                  '2. right elbow',
                  '3. left wrist',
                  '4. right wrist',
                  '5. left wrist']
MAX_N_POINTS = len(body_keypoints)

# --- Load video
video = media.read_video('/path/to/video.mp4')
height, width = video.shape[1:3]
media.show_video(video, fps=30)

select_frame = 0  # @param {type:"slider", min:0, max:49, step:1}

# Generate a colormap with MAX_N_POINTS points
colormap = create_bodypart_colormap(body_keypoints)

selected_points = {}

# --- Open first frame to select points
fig = plt.figure(figsize=(10, 5))
ax_image = fig.add_subplot(121)  # Subplot for the image
ax_list = fig.add_subplot(122)  # Subplot for the list of points
frame = video[select_frame]
ax_image.imshow(frame)  # Display the selected frame from the video
ax_image.axis('off')
ax_image.set_title(
    'Click to select points. \n Spacebar to skip points. \n Afterwards, press enter to close the figure.')
ax_list.axis('off')


# Function to update the list of points
def update_list():
    ax_list.clear()
    ax_list.axis('off')
    for i, keypoint in enumerate(body_keypoints):
        text = keypoint + ': '
        color = colormap[keypoint]
        if keypoint in selected_points:
            text += str(selected_points[keypoint])
        ax_list.text(0, 1 - (i * 0.1), text, va='top', ha='left', fontsize=12, color=color)
    ax_list.figure.canvas.draw()

    # text = part + ': '
    # color = 'black'  # Default color
    # if i < len(select_points):
    #     text += str(select_points[i])
    #     color = tuple(np.array(colormap[i]) / 255.0)  # Set color for the text
    # ax_list.text(0, 1 - (i * 0.1), text, va='top', ha='left', fontsize=12, color=color)


update_list()


def redraw_points():
    """
    Redraws the points on the image.
    """
    global frame

    ax_image.clear()
    ax_image.imshow(frame)  # self.video_data.video[self.select_frame])
    for keypoint in body_keypoints:
        if keypoint in selected_points:
            point = selected_points[keypoint]
            if point is not None:
                color = colormap[keypoint]
                ax_image.plot(point[0], point[1], 'o', color=color)
    ax_image.axis('off')
    update_list()
    plt.draw()


# Event handler for mouse clicks
def on_click(event):
    global selected_points

    if event.inaxes == ax_image:
        x, y = int(np.round(event.xdata)), int(np.round(event.ydata))
        current_keypoint = body_keypoints[len(selected_points)]
        selected_points[current_keypoint] = np.array([x, y])
        redraw_points()
    # if event.button == 1 and event.inaxes == ax_image:  # Left mouse button clicked
    #     x, y = int(np.round(event.xdata)), int(np.round(event.ydata))
    #
    #     # global select_points
    #     select_points.append(np.array([x, y]))
    #     update_list()
    #     # print(select_points)
    #
    #     color = colormap[len(select_points) - 1]
    #     color = tuple(np.array(color) / 255.0)
    #     ax_image.plot(x, y, 'o', color=color, markersize=5)
    #     plt.draw()
    #
    #     # if len(select_points) == 2:
    #     #     fig.canvas.mpl_disconnect(cid)
    #
    #     return select_points


# Event handler for keyboard presses
def on_key(event):
    global selected_points

    # -- Skip points that are not visible with spacebar
    if event.key == ' ':
        current_keypoint = body_keypoints[len(selected_points)]
        selected_points[current_keypoint] = None  # None for skipped points
        update_list()
        # selected_points.append(np.array([None, None]))  # Add an empty point
        # update_list()
        # plt.draw()
    # -- Remove last point with backspace
    elif event.key == 'backspace':
        if selected_points:
            last_keypoint = body_keypoints[len(selected_points) - 1]
            selected_points.pop(last_keypoint)  # Remove the last selected
    # -- Close figure with enter
    # TODO: cannot add more points once size of body_part list has been reached
    elif event.key == 'enter':
        plt.close(fig)

    redraw_points()


cid_mouse = fig.canvas.mpl_connect('button_press_event', lambda event: on_click(event))  # on_click)
cid_key = fig.canvas.mpl_connect('key_press_event', on_key)

plt.show()

print('>>>>>>>>>>>')
print(selected_points)
print('<<<<<<<<<<')
