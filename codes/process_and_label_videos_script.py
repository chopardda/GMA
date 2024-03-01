import os
import sys

current_script_path = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_script_path)

sys.path.append(parent_directory)

from video_manager import VideoManager

video_folder = "/home/daphne/Documents/GMA/data/Preprocessed_Videos"
output_folder = "/home/daphne/Documents/GMA/codes/output"

video_manager = VideoManager()
video_manager.add_all_videos(video_folder, add_pt_data=True)  # Load class data, not the videos themselves

video_ids = video_manager.get_all_video_ids()
# video_ids = ['18_FN_c', '07_F-_c']

for video_id in video_ids:
    print(f"Labelling process for video {video_id}...")
    video_object = video_manager.get_video_object(video_id)

    # -- Load video
    video_object.load_video()

    # -- Label video
    frame_index = 0
    video_object.label_and_store_keypoints(frame_index, task='extreme_keypoints')

    # -- Save labelled points to file
    video_object.save_keypoints_to_csv(os.path.join(output_folder, f'{video_id}.csv'))
    video_object.save_keypoints_to_json(os.path.join(output_folder, f'{video_id}.json'))

    # -- Release video from memory
    video_object.release_video()
