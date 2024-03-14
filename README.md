## Set up environment
Setting up jax to work with GPU is a bit tricky:
1. Create conda environment: `conda env create -f environment.yml`
2. Install jax with GPU support: `pip install --upgrade pip`, `pip install --upgrade "jax[cuda12_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`
3. Install other dependencies: `pip install -r requirements.txt`

Note that the environment variable `XLA_FLAGS=--xla_gpu_force_compilation_parallelism=1` might need to be set, if the CUDA version does not match exactly



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

