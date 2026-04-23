# Gaze4HRI GaT Model Evaluation

## Overview
This repository is a fork of the original GaT repository, specifically adapted for testing on the Gaze4HRI benchmark. For core environment setup and general model information, please refer to the original `README.md`.

## Replication Instructions
Follow these steps to reproduce the gaze estimation results used in the Gaze4HRI paper:

### 1. Environment and Data
* Install the required environment by following the instructions in the original `README.md`.
* Download the Gaze4HRI dataset.

### 2. Configuration
* Create a `.env` file in the root directory by copying the `.env_template`.
* Set the `DATASET_BASE_DIR` variable to your dataset folder path. The directory must contain the date-formatted subdirectories (e.g., `YYYY-MM-DD`).

### 3. Inference
Run the following command to perform inference on all videos in the Gaze4HRI dataset:
`python gaze_estimation_batch.py`

### 4. Export
If you want to export the `gaze_estimations` leaf directories created by `python gaze_estimation_batch.py` under each exp_dir, you can use `flatten_dir.py`.

## Implementation Details
Most Gaze4HRI-related files are copies from the main Gaze4HRI codebase. For detailed information regarding the benchmark infrastructure, refer to the main Gaze4HRI repository.

The unique component of this repository is `gaze_model_gaze3d.py`. This file implements the `GazeModel` abstract class (defined in `gaze_estimation.py`) specifically for the GaT architecture.