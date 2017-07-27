# **Behavioral Cloning**

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # "Image References"

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* behavioral_cloning_8.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model_8_1.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model_8_1.h5
```

#### 3. Submission code is usable and readable

The behavioral_cloning_8.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 or 3x3 filter sizes and depths between 24 and 64 (behavioral_cloning_8.py lines 122-126).

The model includes RELU layers to introduce nonlinearity (code line 122-126), and the data is normalized in the model using a Keras lambda layer (code line 120).

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 12-37). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (behavioral_cloning_8.py line 135).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, several more traning on the dust road and sharp curves.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to make enough convolutional layers followed by several fully connected layers.

My first step was to use a convolution neural network model similar to the LeNet I thought this model might be appropriate because it works fine on previous Traffic Sign Classification project.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set and a low mean squared error on the validation set. This implied that the model was high biasing.

To combat the high biasing, I modified the model so that the model can drive good enough autonomously.

Then I use the Nvidia Network.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I drive the car in trainning mode on these locations for several times to gain more data on these cases.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (behavioral_cloning_8.py lines 118-135) consisted of a convolution neural network with the following layers and layer sizes:

My model consists of a convolution neural network with 5x5 or 3x3 filter sizes and depths between 24 and 64 (behavioral_cloning_8.py lines 122-126).

The model includes RELU layers to introduce nonlinearity (code line 122-126), and the data is normalized in the model using a Keras lambda layer (code line 120).

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I just recorded several drives on the locations where the car can't autonomously pass on road.

To augment the data sat, I also flipped images and angles thinking that this would make the car not to sheer to left too hard and pass through the roads with sharp curve.

After the collection process, I had 43229 number of data points. I then preprocessed this data by cropping the top and bottom part which will distract the model.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I used an adam optimizer so that manually training the learning rate wasn't necessary.
