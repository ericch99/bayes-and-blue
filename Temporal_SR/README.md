# Temporal Super-Resolution


### Purpose 
The goal of this task was to perform video frame interpolation. The original dataset has 6 hours of separation between consecutive images, but we investigate how we can use deep learning -- specifically, a method called "Super SloMo" -- to super-resolve images on the time axis and obtain a clearer picture of how weather patterns change on smaller timescales.

### Notebooks

`baseline_interpolation`: Generate bilinear interpolation baseline. 

`superslomo`: Run Super SloMo on custom dataset (need to specify fps per custom project). 

`similarity_score`: Compute SSIM, PSNR, and VGG loss for image pairs. 

### Sample Videos

Original Video: https://drive.google.com/drive/folders/1J-ejY-BKN7qV-W8VfdY0Ph0lfBisb-Vk?usp=sharing

SuperSlomo Video: https://drive.google.com/drive/folders/1_p5SfPLFxReqbVQTmIAS4-fuYghDv-WP?usp=sharing

### Model Weights

Models: https://drive.google.com/drive/folders/1G1iBfDlNlVpeQKrRbFKDwCHL1zQfxY0T?usp=sharing
