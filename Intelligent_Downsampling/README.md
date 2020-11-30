# Intelligent Downsampling
We use [CycleGAN](https://arxiv.org/abs/1703.10593) to downsample high resolution images to low resolution images, which we then use as training data for our super-resolution task.

### Remarks
The notebook `downsampling.ipynb` contains examples of how to run the training and test code. 
We hosted our code on Google Colab. The notebook contains code to connect to the Google Drive directory and install the missing dependencies. The `checkpoints` folder contains the weights of the trained networks, and sample images generated during training. The `results` folder contains the final generated low resolution images.
