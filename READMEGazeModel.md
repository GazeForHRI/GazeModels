# Gaze4HRI MCGaze Model Evaluation

## Overview
This repository is a fork of the original MCGaze repository, specifically adapted for testing on the Gaze4HRI benchmark. For core environment setup and general model information, please refer to the original `README.md`. However, as mentioned in the main Gaze4HRI codebase, inference with MCGaze was done a bit differently since it was also tested real-time (simultaneously with data collection) during development.

## Replication Instructions
Follow these steps to reproduce the gaze estimation results used in the Gaze4HRI paper:

### 1. Environment and Data
* Install the required environment by following the instructions in the original `README.md`.
* Download the Gaze4HRI dataset.

### 2. Configuration
* Create a `.env` file in the root directory by copying the `.env_template`.
* Set the `DATASET_BASE_DIR` variable to your dataset folder path. The directory must contain the date-formatted subdirectories (e.g., `YYYY-MM-DD`).

### 3. Inference
Inference works based on a UNIX-Domain Socket communication between the main Gaze4HRI repo (the client) and the MCGaze branch of the GazeModels repo (the server).

#### On the MCGaze branch of the GazeModels repository:
Use the following command to run the server that that will receive images from the client and respond with gaze estimations made by the MCGaze model running on the server:
`cd MCGaze_demo && python gaze_server_for_dataset.py`

#### On the main Gaze4HRI codebase:
After the server is running, run the following command to perform inference on all videos in the Gaze4HRI dataset (this is the client that will send images to the server and receive gaze estimations in return):
`python gaze_estimation_batch.py`

### 4. Export
If you want to export the `gaze_estimations` leaf directories created by `python gaze_estimation_batch.py` under each exp_dir, you can use `flatten_dir.py` in the main Gaze4HRI repo.