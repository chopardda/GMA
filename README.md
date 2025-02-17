# Automatic Classification of General Movements in Newborns
This repository includes the instruction and code for the paper "_Towards Scalable Newborn Screening: Automated General Movement Assessment in Uncontrolled Settings_" submitted to ICLR 2025 Workshop AI4CHL.

## Set up environment
Setting up jax to work with GPU is a bit tricky:
1. Create conda environment: `conda env create -f environment.yml`
2. Install jax with GPU support: `pip install --upgrade pip`, `pip install --upgrade "jax[cuda12_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`
3. Install other dependencies: `pip install -r requirements.txt`

Note that the environment variable `XLA_FLAGS=--xla_gpu_force_compilation_parallelism=1` might need to be set, if the CUDA version does not match exactly.



## Keypoint labeling of all videos
The `process_and_label_videos_script.py` script is designed for the sequential labeling of all videos in a given folder, efficiently managing memory by loading and then releasing each video from memory after processing. Labels are generated and saved in both CSV and JSON formats to accommodate different usage scenarios.

### Script Arguments

#### Labelling Task (required)

The `--task` argument specifies the labeling task to be performed and is a required argument for running the script. It determines which keypoints the user will be prompted to label. The available options are:

- `extreme_keypoints`: Use this option for tasks aimed at cropping videos based on specific keypoints. It focuses on labeling extreme keypoints in the video (head top, left elbow, right elbow, left wrist, right wrist, left knee, right knee, left ankle, and right ankle).
  
- `all_body_keypoints`: Select this option when the goal is to extract features from a comprehensive set of body keypoints. This option prompts the labeling of all relevant body keypoints (nose, head bottom, head top, left ear, right ear, left shoulder, right shoulder, left elbow, right elbow, left wrist, right wrist, left hip, right hip, left knee, right knee, left ankle, and right ankle). Note that implementation for `all_body_keypoints` is currently in progress and may not be fully available.


#### Video Source Folder (optional)

- **Path Specification**: The script requires a path to the directory containing the videos to be labeled. This path can be set using the `--video_path` argument.
- **Default Path**: If no path is specified, the script defaults to searching for videos in the `PreprocessedVideos` directory on the cluster.

#### Output Destination Folder (optional)

- **Path Specification**: The output directory for saving the labeled data files (CSV and JSON) can be defined using the `--output_path` argument.
- **Default Path**: By default, labeled files are stored in `./output/labeled`. The script will automatically create this directory if it does not exist.

### Usage Example

To run the script with custom paths for the task `all_body_keypoints` and both the video source and output destination directories, execute the following command:

```sh
python codes/process_and_label_videos_script.py --task all_body_keypoints --video_path /path/to/your/videos --output_path /path/to/save/labels
```

## Outlier detection and removal loop
Removing outliers is an iterative process involving the `identify_outliers_script.py` and `fix_outliers_script.py` scripts. The `identify_outliers_script.py` script helps you to identify the types of outliers in the tracked data, while the `fix_outliers_script.py` script tries to automatically fix as much as it can, and then allows you to relabel the detected outliers. Then, the tracking process is re-run with the supplemented labeled points set. The process is ideally repeated until no more outliers are detected.

### Process
1. Run the `identify_outliers_script.py` script to identify outliers in the tracked data with the following arguments:
`
--show_outliers
--tracked_kp_path /path/to/tracked/keypoint_files
--missing_ok 
--video_path /path/to/your/videos 
--stddev_threshold 20
`
When presented with a potential outlier, use the following keys to label it:
```
escape: Not an outlier
1: Outlier, moving from correct position to incorrect position
2. Outlier, moving from incorrect position to incorrect position
3. Outlier, moving from incorrect position to correct position
```

2. Run the `fix_outliers_script.py` script with the following arguments:
`
--task all_body_keypoints
--tracked_kp_path /path/to/tracked/keypoint_files
--labeled_kp_path /path/to/labeled/keypoint_files
--missing_ok 
--video_path /path/to/your/videos  
--output_path ./output/relabeled
`
For each frame that pops up, relabel the outlier point, as well as any other incorrect points as needed. To do this, click on the point to be relabeled, and then click on the correct point in the video. Press enter when finished with that frame.
3. Delete the `./output/outliers` directory and the `./keypoints_distributgions.pkl` file.

4. Rerun the tracking process, using the relabled data as labeled input. e.g.:
`
python main.py --task all_body_keypoints --video_path /path/to/your/videos  --labeled_kp_path ./output/relabeled --tracked_kp_path ./output/tracked_relabeled --missing_ok --track_only
`
5. Rerun steps 1-4, using `./output/relabeled` as the labeled_kp_path and `./output/tracked_relabeled` as the tracked_kp_path, and so on.
6. After you are satisfied, the final tracked keypoints located at the path provided to the `--tracked_kp_path` flag in the last iteration can be used for further analysis.


## Classification problem from tracked keypoints

Computing the label prediction for the classification problem in the tracked videos. Using the coordinates of the keypoints and cross validation. Execute the following command:

`
python feature_extraction.py 
`

You can modify the parameters for the training with the following flags:
- `--seed`: Random seed
- `--epochs`: Number of epochs
- `--folds`: Number of folds for cross validation
- `--batch_size`: Batch size
- `--type_a`: Type of classification problem between the early and late videos. Options: `early` or `late`
- `--directory`: Path to the tracked keypoints
- `--model`: Model architecture to use for training
- `--wandb`: Save logs to cluster local WandB instance. It must be set up correctly
- `--num_iterations`: Number of full cross validation iterations to run
- `--label_method`: Type of keypoint label extraction. Options: `aggpose` or `None`