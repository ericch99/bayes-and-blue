# Detection of Extreme Weather Phenomena

## Purpose
The goal of this task was to validate the SRGAN output by comparing object detection accuracy between different types of images: ground-truth high-resolution images, low-resolution images upsampled using bicubic interpolation, and low-resolution images upsampled using the SRGAN. We chose to use a single-stage detector, [YOLO](https://arxiv.org/abs/1506.02640), to accomplish this due to its computational efficiency.

Each notebook contains code for setting up, training, and evaluating the YOLO model on the respective version of the ExtremeWeather dataset and should work mostly out-of-the-box; the main things to change are the file paths for data loading as well as model hyperparameters, if desired. The cells can simply be run in order.

## Acknowledgments 
These notebooks were adapted from the University of Michigan course EECS 442: Computer Vision (PS7). 
