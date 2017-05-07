#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a 3 convolutional layers and 2 fully connected layers. The convolutional layers have a SAME padding, elu activation and 4 by 4 stride.

My fully connected layers have elu activation as well.


####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. The first fully connected layer has a dropout of 0.5.

I originally trained with 22 epochs. This was clearly too high, as my loss was oscilating back and forth rather tahn gradually decreasing. This indicated that my model was overfitting. I eventually settled on an epoch number of 5.

####3. Model parameter tuning

I used an Adam optimizer.

####4. Appropriate training data

I used the training data provided by Udacity. I also tried to gather my own, however, there were times where I couldn't quite steer the car simulation correctly. I did not want to introduce bad driving data to my model, so I did not use my own data.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to add 3 convolutional layers. After a bit of googling around, I learned that convolutions work well with car simulated image data.

I split my data into a training set and validation set.

For the first convolutional layer, I used 16 filters, 8 * 8 kernals, 4 * 4 strides, and same padding and elu activation
For the second convolutional layer, I used 32 filters, 5 * 5 kernals, 2 * 2 strides, and same padding and elu activation
For the third convolutional layer, I used 62 filters, 5 * 5 kernals, 2 * 2 strides, and same padding and elu activation

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

####3. Creation of the Training Set & Training Process

Pre-processing of input data

After a big of research, I cropped the images. I realized that not all parts of the image are necessary for my model, or have anything to do with driving/steering.
I also have a habit of normalizing the data for a cleaner convergence so I did that.
I made the images smaller to simplify the data.
Lastly, I shuffled the data.
95% of the data went into the validation set.

At first, I used only the centre camera images. This did not result in a very good driving simulation, as my car eventually ended up veering off to the left. To correct this, I took all three camera images, and added a correction of 0.2.

The validation set was to diagnose my model, particularly tell whether it had high bias (underfitting) or high variance (overfitting)

I found that 5 epochs was ideal.

####Images - My dataset
The image at the path images/center.jpg is centre camera image, with a steering of 0
The image at the path images/left/jpg is a left camera image, with a steering of 0
The image at the path images/right.jpg is a right camera image, with a steering of -0.0787459

