#**Behavioral Cloning Project** 
**By Jaeyong Han, 20/Mar/2017**


## Project Description
 This project has the goal to mimic human driving behavior with Convolutional Neural Network using data from the simulator which is provied by Udacity as Self Driving Car nanodegree program.
 The steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Adjust Steering wheel angle
* Image Preprocessing (crop)
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./image/model.png "Model Visualization"
[image2]: ./image/center.jpg "Grayscaling"
[image3]: ./image/left.jpg "Recovery Image"
[image4]: ./image/right.jpg "Recovery Image"
[image5]: ./image/center.jpg "Recovery Image"
[image6]: ./image/center.jpg "Normal Image"
[image7]: ./image/flipped.jpg "Flipped Image"

## Rubric Points
###Files Submitted & Code Quality
###1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model_trk1_n3.h5 containing a trained convolution neural network for track 1 with ONLY track 1 data
* model_trk2_n3.h5 containing a trained convolution neural network for track 2 with ONLY track 2 data
* model_trk12_n3.h5 containing a trained convolution neural network for both tracks with BOTH data
* video_trk1.mp4 containing a video from model model_trk1_n3.h5(ONLY from track 1 data)
* video_trk2.mp4 containing a video from model model_trk2_n3.h5(ONLY from track 2 data)
* video_trk12.mp4 containing a video from model model_trk12_n3.h5(2 tracks from BOTH tracks data)
* writeup_JYHan.md summarizing the results

###2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model_trk12_n3.h5
```

###3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

###1. An appropriate model architecture has been employed

My model is based on NVIDIA model which has 5 Convolution layers and 3 fully connected layers.(Code line 109-123)
The model includes RELU layers to introduce nonlinearity (code line 112-116), and the data is normalized in the model using a Keras lambda layer (code line 110). 

###2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 118,121). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 55). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

###3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 125).

###4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and example data which is alread given.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

###1. Solution Design Approach
I approached the solution with following 2 principles:
* Use introduced model in the lecture which was from the paper by NVIDIA.
  because its performace was verified with many trials. So, the introduced model with handling parameters is more effective than using other model or creating new model.
* The key of ensuring model accuracy is data.
  Data should simulate real driving patterns. But, With controling keyboard, data has unstable steering wheel angle.
  For example, in human drvining, we keep estimating steering wheel angles continuously. In simulator, I keep adjusting positions using pushing arrow keys(descrete). It make huge differences during modeling. 

The introduced model in the lecture is verified model by NVIDIA. and I planned to add more depths if I cannot get significant accuracy on model with handling overfitting using dropout. and, fortunately, It shows quite enough accuracy to apply autonomous driving in the simulator.
 while using the NVIDIA model, I adjusted several factors to have better model.
 * Dividing Training and Validation Set, I started my modeling with 8:2 on training vs validation dataset. But, I had more data from additional trails on simulator. I changed the proportion to 7:3
 * Change number of itration of epoches with the trends of epoches to find the ideal number.

and during modeling with additional data, my model brought me lower accuracy then before adding data. and I found the autonomous vehicle in simulator fell off the track with additional data.
 So, I focused on quality of data. and I made an assumption that all this unstability on model are from the value of steering wheel which I made by my keyboard. It isn't same when I handle my steering wheel in my car. In real, It should be moved continuously. But, In simulator, It was discrete with pushing button to adjust position vehicle.
 
 With adjust the value of steering wheel, the vehicle is able to drive autonomously around the track without leaving the road.

###2. Final Model Architecture

The final model architecture (model.py lines 109-123) consisted of a convolution neural network with the following layers. I used 5 convolution layers, 1 flatten, 3 fully connected layers and 2 dropout layers

Here is a visualization of the architecture.

![alt text][image1]

###3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to how to be back the vehicle to the center of the path. These images show what a recovery looks like starting from left image, right image and centered image. :

![alt text][image3]
![alt text][image4]
![alt text][image5]

and for more data, I added example data for the model

Then I repeated this process on track two in order to get more data points. but, I needed to spend more time to control the vehicle in simulator due to rapid curves.

To augment the data set, I also flipped images and angles thinking that this would provide more data with new experiences. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

After the collection process, I had 132,186 number of data points for track 1 and 141,180 number for track2. 

To simulate real driving behavior, I needed to adjust the angle of steering wheel. the basic idea is explained in "Solution Design Approach" Section. But, if I say it shortly again, the idea is, in real driving, we can see the path to drive and we are handling the steering wheel for curve just before entring to curve area. So, I made average of 3 in-advanced data including the current value of steering wheel. and this treatment gave me more stable control in the simulator.

I finally randomly shuffled the data set and put 30% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 50 as evidenced by lower loss on 50. I used an adam optimizer so that manually training the learning rate wasn't necessary.

###3. Trial for Track 2

I made 3 models for track1, track2 and both(works for both tracks). model for track1 and 2 work well. and A model for both tracks is working as well. But, sometimes, It drove to the side of road. but not crashed. finished at least a full track of each track.
