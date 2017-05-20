# **Traffic Sign Recognition** 

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

### Data Set Summary & Exploration

#### 1. Basic summary of the data set. 

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset.

See code under 'Include an exploratory visualization of the dataset' section

### Design and Test a Model Architecture

#### 1. Preprocess the image data.

I decided to normalize the images for a cleaner optimization, i.e it allows you to reach a convergence faster. 

See 'Pre-process the Data Set (normalization, grayscale, etc.)' for an example of an image before and after normalization.

I made a decision to not greyscale. Colors are quite important here. Unlike lane lines, a traffic sign's meaning is derived from not only just its shape, but also its color.


#### 2. Final model architecture (model type, layers, layer sizes, connectivity, etc.) 
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


#### 3. Training the model, including type of optimizer, the batch size, number of epochs and hyperparameters 

My batch size and epochs were a result for trial and error. I found that a slightly higher number of epochs did not really yield that much better of accuracy (sometimes even less) for the extra time taken. Also, a smaller/larger batch size would yield a lower accuracy as well. 

My learning rate did not require any tweaking. After a bit of googling around and reading, I decided to come to a value of 0.001, and this worked out just fine.

#### 4. validation set accuracy is at least 0.93. 

My final model results were:
* training set accuracy of ～80%
* validation set accuracy of ~96%
* test set accuracy of ~85%

At first I used the tf.train.GradientDdescentOptimizer. However I encountered 2 problems. 
1. Reaching convergence was very slow
2. The accuracy was quite low (~50%)

After a bit of Googling around, I decided to use the tf.train.AdamOptimizer. This converged to an optimal solution much quicker and yeilded a higher accuracy.  
 

### Test a Model on New Images

#### 1. 5 German traffic signs found on the web.

Please refer to the traffic-signs-data folder for the 5 German images.

The images named german2.jpg & german4.jpg would be difficult to classify because of the watermarks.
The image named german3.jpg is not a very common sign. Again, this may be difficult to classify.
The image named german5.jpg appears to be an image of a boat. I couldn't see anything in signnames.csv related to this, so I was curious to see what my model would come up with.


#### 2. Model's predictions on these new traffic signs 

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (70km/h)  | Keep right   									| 
| Exlamation mark , caution? | Yield 										|
| People on phone		 Go straight or left											|
| Yield	      		|    Road narrows on the right					 				|
| Symbol of a boat?		| Priority road      							|


The model was able to correctly guess 0 of the 5 traffic signs, which gives an accuracy of 0%. This is signficantly less than the accuracy on the test set of ~85%

#### 3. Certainty of the model when predicting on each of the five new images 


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Kepe right   									| 
| 1     				| Yield 										|
| 1					| Go straight or left											|
| 1	      			| Road narrows on the right					 				|
| 1				    | Priority Road      							|



Despite being wrong 5 times in a row, the model was quite certain about all of its predictions. A confident idiot ;)
