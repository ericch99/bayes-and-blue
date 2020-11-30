# Intelligent Downsampling
We use CycleGAN to downsample high resolution images to low resolution images, which are then used as training data for our super-resolution task. 
The notebook `downsampling.ipynb` contains examples of how to run the training and test code. 
We hosted our code on Google Drive and ran the code on Google Colab, thus the notebook contains code to connect to the Google Drive directory and install the missing dependencies. The `checkpoints` folder contains the weights of the trained networks, and sample images generated during training. The `results` folder contains the final generated low resolution images.
