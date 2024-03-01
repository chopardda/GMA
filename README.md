## Label all videos
Use the script `codes/process_and_labels_videos_script.py` to label all videos one after the other. The script load and release each video in order to lower the memory requirements. Labels are saved both as 'csv' and 'json' files. 
Note: Before using the script, the correct path to `video_folder` (where all videos to labels are stored) and `output_folder` (where the labelled points files should be stored) need to be set.

For now, the labelling task has been set to `extreme_keypoints` in order to crop the videos accordingly. However, the task `all_body_keypoints` should be used when the goal is to extract features. This second option has not been fully implemented yet.