# Image Colorization Starter Code
The objective is to produce color images given grayscale input image. 

## Setup Instructions
Create a conda environment with pytorch, cuda. 

`$ conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia`

For systems without a dedicated gpu, you may use a CPU version of pytorch.
`$ conda install pytorch torchvision torchaudio cpuonly -c pytorch`

## Dataset
Use the zipfile provided as your dataset. You are expected to split your dataset to create a validation set for initial testing. Your final model can use the entire dataset for training. Note that this model will be evaluated on a test dataset not visible to you.

## Code Guide
Baseline Model: A baseline model is available in `basic_model.py` You may use this model to kickstart this assignment. We use 256 x 256 size images for this problem.
-	Fill in the dataloader, (colorize_data.py)
-	Fill in the loss function and optimizer. (train.py)
-	Complete the training loop, validation loop (train.py)
-	Determine model performance using appropriate metric. Describe your metric and why the metric works for this model? 
- Prepare an inference script that takes as input grayscale image, model path and produces a color image. 

## Additional Tasks 
- The network available in model.py is a very simple network. How would you improve the overall image quality for the above system? (Implement)
- You may also explore different loss functions here.

## Bonus
You are tasked to control the average color/mood of the image that you are colorizing. What are some ideas that come to your mind? (Bonus: Implement)

## Solution
- Document the things you tried, what worked and did not. 
- Update this README.md file to add instructions on how to run your code. (train, inference). 
- Once you are done, zip the code, upload your solution.  

# Solution

The report about the experiments is in report.ipynb run 
`jupyter notebook`
to check it

## Training:

`virtualenv venv`

`source venv/bin/activate`

`pip install -r requirements.txt`

and then
(for lab format)

`cd lab`

`python run_lab.py PATH_WITH_IMAGES [args]`

or (for rgb format)

`cd rgb`

`python run_rgb.py PATH_WITH_IMAGES [args]`

the models will be saved in models directory and the details about the training (including sample images)
will be uploaded in real time to 
https://app.neptune.ai/rm360179/image-coloring

## Interference:

To run the previously trained model, run
`python interference.py img_path model_path ouput_path`
in rgb or lab directory

They sample images are much easier to see at neptune website though
