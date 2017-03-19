# **Behavioral Cloning** 

## Project Summary

In this project, I used the provided simulator to drive around a track to collect training data demonstrating good driving behavior.  I then built
a convolution neural network in Keras that would predict the desired steering angle based on an input image.  This model was then trained with the
training data from my driving around the track.  The simulator has another mode that can be used to test the model by feeding images from the track
into the model and using the computed steering angle from the model to drive the car.  I have included a video of the model successfully driving
around the track.

One of the reasons this project is difficult is that the model needs to produce a steering angle from a single frame image.  There is no situational
awareness about the vehicle's offset from the center of the lane or information about previous angles or movement of the vehicle.  Therefore the model 
needs to detect the relevant visual features (lane lines) from the frame and determine the appropriate action using only the pixels in a single image.
This is different from the approach in the "Finding Lane Lines" project where information from multiple frames was averaged to smooth out variations 
in the detection of lane lines.

[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./images/center_2017_03_13_21_08_19_958.jpg "center_2017_03_13_21_08_19_958.jpg"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

### Files Submitted &amp; Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md this writeup
* video.mp4 video of model driving one complete lap of the first track

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and 
validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I chose to use adapt the neural network developed by Nvidia in End-to-End Deep Learning for Self Driving cars [link](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/).
I my testing this model gave good performance with Behavioral Cloning and was responsive to new training data.
My model is defined in model.py lines 52-67.  It consists of 5 convolutional layers, a flatten layer and then 4 fully-
connected layers.

Each convolutional layers utilizes a RELU activation function to introduce nonlinearity (code line 58-62), 
and the data is normalized in the model using a Keras lambda layer (code line 55).

In addition to the normalization, I also used the Keras Cropping2D filter to focus the network on the bottom portion of the 
image.  This is primarily where the road information comes from and helped cut out noise from the sky and background that
we don't want the model to train on.

#### 2. Attempts to reduce overfitting in the model

I used training data from multiple runs around the track, including running backwards around the track to try and prevent overfitting.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was recorded by my driving around the track in the simulator.  To begin, I drove carefully around the track
and then added additional images from difficult portions of the track and recovery from incorrect vehicle/track alignment.
I manually edited the csv driving log file, comparing recording steering angles and images to assemble my final set of
training data, including data taken from multiple different recorded sessions.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My approach was to incrementally improve both my training data set and my model until the model could successfully drive itself
around the first track.

I first started with the LeNet-5 network and training data from a single lap around the track.  I found that the LeNet network
did a reasonable job of training on the data, but that the combined error when testing it was too high and the vehicle would
eventually fall off of the edge of the track.  I then moved on to using the more sophisticated model from Nvidia.

I began with the model taken from the article with the count of the neurons in the final fully-connected/Dense layers being
1164, 100, 50, 1.  I found by experimentation that having 50 neurons in the second-to-last layer (model.py, line 65), made
the car steer back and forth in the lane a lot.  By reducing the count of neurons in that layer to 10, I observed smoother
driving in the simulator.

My car was still not able to make it completely around the track, so I began by recording additional training images for sections
of the track that the model was struggling with.  Normally, this might contribute to "over-fitting," but in this case the goal
was to achive Behavior Cloning, so I thought that providing more examples of my driving was acceptable.

In addition to center-lane driving through difficult sections, I recorded a series of recoveries, where the vehicle would begin by
facing towards one side of the lane or the other and then I would record the steering angles required to bring it back into
the center of the lane.  This helped my model drive around the track tremendously.

I also added another complete forward and backwards lap to round out the training data set.

Once I had added all of those images, I had 7,406 frames of training data.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 58-67) consisted of a convolution neural network adapted from Nvidia.  I found that
once I had a larger test data set I was able to increase the number of neurons in the second-to-last layer from 10 to 14 and
that resulted in slightly more searching, but overall more accurate driving from the model.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one lap on track one using center lane driving. 
Here is an example image of center lane driving.

![alt text][image2]

Based on comments from other students on the forums, I deleted portions of my first lap that were straight to reduce straight
bias in my model.

To augment the data sat, I also flipped the center image and steering angle to help prevent the left bias from driving around
the mostly-circular track.

I also added the images from the left and right cameras with an slight adjustment to the steering angle.  This helps the model learn
how to recover when it is off center in the lane.

I eventually added training images from driving around track one backwards, but I left the data flipping augmentation in because
it had the nice side effect of having the model train on an equal number of center-camera images as side-camera images.  I like
this because for the simulation, the model is only fed center camera images, so I want to be able to predict steering angle
primarily from this perspective.  I could have also dropped 50% of the side-camera images.

I used the shuffle and validation_split parameters of Keras model's fit function to randomly shuffle the images used for training
and reserve 20% of the data for validation.  This is another way to help prevent over-fitting in the model.

When I was training on my local computer, I used the Keras EarlyStopping callback to stop training when the rate of improvement
in the validation loss began to decrease.  I found that the loss started to flatten out very quickly, usually after 2 or 3 Epocs.
Eventually I moved to training the model on AWS and I removed this callback and just trained for 10 epocs.
