## Label all videos
Use the script `codes/process_and_labels_videos_script.py` to label all videos one after the other. The script load and release each video in order to lower the memory requirements. Labels are saved both as 'csv' and 'json' files. 
Note: Before using the script, the correct path to `video_folder` (where all videos to labels are stored) and `output_folder` (where the labelled points files should be stored) need to be set.

For now, the labelling task has been set to `extreme_keypoints` in order to crop the videos accordingly. However, the task `all_body_keypoints` should be used when the goal is to extract features. This second option has not been fully implemented yet.

## Set up environment
Setting up jax to work with GPU is a bit tricky:
1. Create conda environment: `conda env create -f environment.yml`
2. Install jax with GPU support: `pip install --upgrade pip`, `pip install --upgrade "jax[cuda12_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`
3. Install other dependencies: `pip install -r requirements.txt`