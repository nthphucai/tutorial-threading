# Tutorial - Multithreading  

## Overview

This README provides an overview of a multithreading pipeline composed of three components: `camera_handler` (B1), `pose_handler` (B2), `violence_handler` (B3). These components work together to process images from a camera, extract poses, analyze violence probability, and draw the results.

### Components:

- **B1: camera_handler**
  - Reads images from the camera.
  - Puts each image into `image_queue` without waiting.
  - Checks if `image_queue` is full; if so, retrieves an image using `image_queue.get()` and checks `image_condition`.
  - Notifies all threads when `image_queue` is not empty.
  - Checks if `draw_queue.qsize()` true, draw images and skeleton.

- **B2: pose_handler**
  - While `isPlaying` is True, checks `image_condition`.
  - If `image_queue` is empty, waits. If `image_queue.qsize()` true, processes and puts frames into `pose_queue`.
  - Checks if `pose_queue` is full; if so, retrieves a pose using `pose_queue.get()`.
  - Checks `pose_condition` and sends a notification to `violence_handler` when `pose_queue` reaches batch size.
  - Processes images and puts them into `draw_queue` without waiting.
  - Checks if `draw_queue` is full; if so, retrieves a frame using `draw_queue.get()`.

- **B3: violence_handler**
  - While `isPlaying` is True, checks `pose_condition`.
  - Waits if `pose_queue` is not enough to reach batch size.
  - If `pose_queue` reaches batch size, calculates violence probability.

## Prerequisites
- Python 3.8+
- Libraries/packages can be installed by using requirements.txt file:

```python
pip install -r requirements.txt
```
## CUDA Requirements

The following components in this pipeline leverage GPU acceleration through CUDA. Ensure that you have CUDA installed on your system before running the pipeline. You can install CUDA by following the instructions on the [NVIDIA CUDA Toolkit website](https://developer.nvidia.com/cuda-toolkit).

## Models

### 1. Violence Detection Model

The violence detection model is responsible for analyzing pose data and predicting the likelihood of violence. The model is trained to recognize patterns indicative of violent actions. [Violence-Detection-Model-Link](https://drive.google.com/drive/u/0/folders/1-8gadhTheB_ZdUa3nAu_WakCiU8c-mbS)

### 2. YOLO-based Pose Estimation Model

The YOLO-based pose estimation model extracts human poses from video frames. It plays a crucial role in providing input data for the violence detection model. [Yolo-Model-Link](https://drive.google.com/drive/u/0/folders/1-8gadhTheB_ZdUa3nAu_WakCiU8c-mbS)

## Usage
Run the script using the following command:

```python
python violence_detection/detect_violence.py --debug False --output_path=output_video.avi --source=test_video.avi
```
Optional arguments:

--debug: Enable debug mode (default: True).

--output_path: Specify the output path for the result video (default: "output_video.avi").

--source_id_or_video_path: Specify the source ID or video path for analysis (default: "test_video.avi"). 

### Download Test Video

You can download the `test_video.avi` file from the following link:
[Download Test Video](https://drive.google.com/file/d/1BadglD07Anozu6lx4xl-VoX1tgpxxOkm/view?usp=sharing)

Note: If you want to record real-time video from the camera, you can search online to find instructions on how to check your camera ID. The default camaera ID is 0. 

## Notes

- Carefully manage thread synchronization using conditions and queues to prevent race conditions and ensure smooth communication between components.
- Adjust parameters such as batch size and queue capacities to optimize performance based on system resources and specific requirements.


## Contribution
Feel free to contribute by enhancing the pipeline's functionality, improving performance, or providing additional documentation. Open issues and submit pull requests as needed.


## Author
Full name: Nguyen Thi Hong Phuc

Email: nthphucai@gmail.com

# Acknowledgment
The pose estimation model is based on YOLOv7.

Implementation for Violence-Detection built upon the following work:
```
@article{GARCIACOBO2023SkeletonsViolence,
  title = {Human skeletons and change detection for efficient violence detection in surveillance videos},
  journal = {Computer Vision and Image Understanding},
  volume = {233},
  pages = {103739},
  year = {2023},
  issn = {1077-3142},
  doi = {https://doi.org/10.1016/j.cviu.2023.103739},
  url = {https://www.sciencedirect.com/science/article/pii/S1077314223001194},
  author = {Guillermo Garcia-Cobo and Juan C. SanMiguel}
}
Feel free to explore and modify the code for your own projects! If you have any questions or issues, please contact me via email: nthphucai@gmail.com
```