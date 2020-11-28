# YOLO Object Detection Model
Bounding Box Detection for Validation of Image Super-Resolution

## Purpose
The goal of this task was to validate the SRGAN output by comparing object detection accuracy between different types of images: ground-truth high-resolution images, images that had been downsampled 8 times and then upsampled using bicubic interpolation, and images that had been downsampled 8 times and then upsampled using the SRGAN. 

Each separate notebook contains code for setting up, training, and evaluating the YOLO model on the ExtremeWeather dataset and should work mostly out-of-the-box; the main things to change are the file paths for data loading as well as model hyperparameters, if desired. 

## Acknowledgments 
These notebooks were adapted from the University of Michigan course EECS 442: Computer Vision (PS7). 
