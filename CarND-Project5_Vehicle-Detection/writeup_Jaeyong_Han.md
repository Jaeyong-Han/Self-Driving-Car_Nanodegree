##Vehicle Detection and Tracking
---
Created by Jaeyong Han

###Project Overview

The purpose of this project is to write a software pipeline to identify vehicles in a video from a front-facing camera on a car. 

The goals / steps of this project are the following:

1st Phase : Build a classifier from Car / Non-Car images
* Generate more dataset using gamma correction to mimic different light condition.
* Test all color spaces and color channels to find best for Histogram of Oriented Gradients (HOG) with linear SVM
* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear or Non-linear SVM classifier per each color spaces which are seleted from previous test.
* Take the best performance classifier from the result of feature extraction and trained classifier
(!) Dataset is shuffled before spliting dataset into training/test dataset and during spliting dataset, data is ramdom sampled.
(!) Before feed features to training/predict processes, features are normalized.


2nd Phase : Find vehicle using sliding window technique with trained Classifier.
* Implement a sliding-window technique and use the best trained classifier to search for vehicles in images.
* Run pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./writeup_images/car_not_car.png
[image2]: ./writeup_images/gamma_correction.png
[image3]: ./output_images/test3_out.jpg
[image4]: ./output_images/test1_oout.png
[image5]: ./output_images/test1_heat.png
[video1]: ./output_images/project_video_out.mp4

###Project Explanation 
There are [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points which I must consider for the project.

#####Rubric Points for Histogram of Oriented Gradients (HOG)

######1. Explain how (and identify where in your code) you extracted HOG features from the training images and how you settled on your final choice of HOG parameters.

First, I loaded images about the `vehicle` and `non-vehicle` from files which are provided for the project.(7th cell in `Parameter_research.ipynb`)
Here is a pair of pictures about the `vehicle` and `non-vehicle`
![alt text][image1]

and I created more images using gamma corrections(0.5,2) (4th cell in `Parameter_research.ipynb`)
![alt text][image2]

After loading the `vehicle` and `non-vehicle` dataset, I made a strategy how to compare among color spaces and HOG parameters.
I decided to use a linear SVM classifier for each cases and compare accuracy from each cases.

Here is the accuracy comparison table for color spaces with HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`which I found the best parameters for HOG.


| Color Space   | RGB   |  HSV  |  LUV  |  HLS  |  YUV  | YCrCb |  YUL  |
|:-------------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:| 
| All Channel   | 0.96  | 0.97  | x     | 0.97  | x     | 0.98  | 0.98  | 
| Channel 1     | 0.93  | 0.90  | 0.93  | 0.90  | 0.94  | 0.94  | 0.94  |
| Channel 2     | 0.94  | 0.89  | x     | 0.93  | 0.93  | 0.92  | 0.93  |
| Channel 3     | 0.94  | 0.93  | x     | 0.89  | x     | 0.90  | 0.93  |

(!) YUL : New color space YU from YUV and L from HLS
(!) To calculate accuracy for each color space, I used search_classify.py from lecture with changing color_space variale and HOG parameters.


######2. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

To train classifier, I chose SVM with linear and radial(rbf) and I took following steps.
1) Data standardization.
2) shuffle data. 
3) split up date into training and test sets(Random Sampling)
4) SVM parameter estimation using GridSearch
5) train classifier.
(6th cell, function modeling in `Parameter_research.ipynb`)

Radial model with {'kernel': 'rbf', 'C': 10, 'gamma': 0.0001} gives me more higher accuracy than linear model. But, too much time spending to train classifier and prediction for project videos. So, I use linear model which has abit low accuracy than radial model. But. Good performance under considering with processing time.

#####Sliding Window Search

######1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used simple wisdom. 'Many trials, Good result'. Scales are from 0.5 to 2 with 0.2 steps. Totally, 8 scales. and the window moves 2 cells per step.
Sliding window search is coded on 15~77 in find_object.py


######2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I searched on 8 scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided quite good result.  Here are an example image:

![alt text][image3]
---

##### Video Implementation

######1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_images/project_video_out.mp4)


######2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video. (coded on 132~142 in find_object.py)

Here are an image with rounding box and its corresponding heatmap:
![Rounding Box][image4]
![Heatmap][image5]


---

#####Discussion

######1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Most critical issue is model is too heavy to apply for real-time processing. Decision Tree(or Random forest) might bring similar but fast result. and I searched with various scales and it makes slowness during processing. In future, I will find some adequated scale and apply them based on distance.

