# **Behavioral Cloning** 

## Final output

![demo](run2.gif)

## Images - My dataset

centre camera image, with a steering of 0.

![alt text][image1]

 left camera image, with a steering of 0.
 
![alt text][image2]

right camera image, with a steering of -0.0787459.

![alt text][image3]


### Model Architecture and Training Strategy

#### 1. Model architecture

My model consists of a 3 convolutional layers and 2 fully connected layers. 

The convolutional layers have a SAME padding & elu activation

My fully connected layers have elu activation as well.


#### 2. Reduce overfitting 

The model contains dropout layers in order to reduce overfitting. The first fully connected layer has a dropout of 0.5.

I originally trained with 22 epochs. This was clearly too high, as my loss was oscilating back and forth rather than gradually decreasing. This indicated that my model was overfitting. I eventually settled on an epoch number of 5.

#### 3. Model parameter tuning

I used an Adam optimizer.

#### 4. Appropriate training data

I used the training data provided by Udacity. I also tried to gather my own, however, there were times where I couldn't quite steer the car simulation correctly. I did not want to introduce bad driving data to my model, so I did not use my own data.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to add 3 convolutional layers. After a bit of googling around, I learned that convolutions work well with car simulated image data.

I split my data into a training set and validation set.

For the first convolutional layer, I used 16 filters, 8 X 8 kernals, 4 X 4 strides, and same padding and elu activation
For the second convolutional layer, I used 32 filters, 5 X 5 kernals, 2 X 2 strides, and same padding and elu activation
For the third convolutional layer, I used 62 filters, 5 X 5 kernals, 2 X 2 strides, and same padding and elu activation

#### 2. Final Model Architecture

Here is a code snippet of my final model architecture:
```python
#crop
model.add(Cropping2D(cropping=CROPPING_DIMS, 
                     dim_ordering='tf',  
                     input_shape=INPUT_SHAPE)) 

# normalize
model.add(Lambda(lambda x: (x/255.0) - 0.5))

# add convolutional layers 
model.add(Convolution2D(*CONV_1, subsample=(4, 4), border_mode="same")) 
model.add(ELU()) 
model.add(Convolution2D(*CONV_2, subsample=(2, 2), border_mode="same")) 
model.add(ELU()) 
model.add(Convolution2D(*CONV_3, subsample=(2, 2), border_mode="same")) 
model.add(Flatten()) 
model.add(ELU()) 

# add fully connected layers
model.add(Dense(FULLY_CONNECTED_LAYER1)) 
model.add(Dropout(DROPOUT)) 
model.add(ELU()) 
model.add(Dense(FULLY_CONNECTED_LAYER2)) 
model.add(ELU()) 
model.add(Dense(FULLY_CONNECTED_LAYER3)) 
```

#### 3. Creation of the Training Set & Training Process

Pre-processing of input data

After a big of research, I cropped the images. I realized that not all parts of the image are necessary for my model, or have anything to do with driving/steering.
I also have a habit of normalizing the data for a cleaner convergence so I did that.
I made the images smaller to simplify the data.
Lastly, I shuffled the data.
95% of the data went into the validation set.

At first, I used only the centre camera images. This did not result in a very good driving simulation, as my car eventually ended up veering off to the left. To correct this, I took all three camera images, and added a correction of 0.2.

The validation set was to diagnose my model, particularly tell whether it had high bias (underfitting) or high variance (overfitting)

I found that 5 epochs was ideal.

[//]: # (Image References)

[image1]: ./images/center.jpg 
[image2]: ./images/left.jpg 
[image3]: ./images/right.jpg 

## Final output

![demo](run2.gif)
