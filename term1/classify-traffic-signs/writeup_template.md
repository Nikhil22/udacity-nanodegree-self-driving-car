#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

See code under 'Include an exploratory visualization of the dataset' section


###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I decided to normalize the images for a cleaner optimization, i.e it allows you to reach a convergence faster. 

See 'Pre-process the Data Set (normalization, grayscale, etc.)' for an example of an image before and after normalization.

I made a decision to not greyscale. Colors are quite important here. Unlike lane lines, a traffic sign's meaning is derived from not only just its shape, but also its color.


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolutional layer 1     	| [1,3,3,1] stride, valid padding, outputs 5x5x32, 32 filters 	|
| 					|												|
| Max pooling	      	| strides [1,2,2,1], valid padding ' 				|
| Convolution 3x3	    | etc.      									|
| Fully connected layer 1		| dropout @ 0.85,  relu    									|
| Softmax				| etc.        									|
|fully connection layer 2	 dropout @ 0.85, relu 				|												|
|						|												|
 
 
 Reference: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network.py


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ～80%
* validation set accuracy of ~96&
* test set accuracy of ~85%

At first I used the tf.train.GradientDdescentOptimizer. However I encountered 2 problems. 
1. Reaching convergence was very slow
2. The accuracy was quite low (~50%)

After a bit of Googling around, I decided to use the tf.train.AdamOptimizer. This converged to an optimal solution much quicker and yeilded a higher accuracy.  
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Please refer to the traffic-signs-data folder for the 5 German images.

The images named german2.jpg & german4.jpg would be difficult to classify because of the watermarks.
The image named german3.jpg is not a very common sign. Again, this may be difficult to classify.
The image named german5.jpg appears to be an image of a boat. I couldn't see anything in signnames.csv related to this, so I was curious to see what my model would come up with.


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (70km/h)  | Keep right   									| 
| Exlamation mark , caution? | Yield 										|
| People on phone		 Go straight or left											|
| Yield	      		|    Road narrows on the right					 				|
| Symbol of a boat?		| Priority road      							|


The model was able to correctly guess 0 of the 5 traffic signs, which gives an accuracy of 0%. This is signficantly less than the accuracy on the test set of ~85%

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Kepe right   									| 
| 1     				| Yield 										|
| 1					| Go straight or left											|
| 1	      			| Road narrows on the right					 				|
| 1				    | Priority Road      							|



Despite being wrong 5 times in a row, the model was quite certain about all of its predictions. A confident idiot ;)
