##Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./camera_cal/calibration1.jpg "Distorted"
[image2]: ./camera_cal/undist1.jpg "Undistorted"
[image3]: ./output_images/und_st_lines1.jpg "Road Transformed"
[image4]: ./output_images/warp_straight_lines1.jpg "Warp Transformed"
[image5]: ./output_images/prep_straight_lines1.jpg "Color Trabsformed"
[image6]: ./examples/color_fit_lines.jpg "Fit Visual"
[image7]: ./output_images/output_Image.png "Output"
[video1]: ./output_images/res_project_video.mp4 "Video"

Strategy to complete Project 4:
* Camera Calibration and Source/Destination points for perspective transform generates first as parameters for universal use in project 4
* Each parts should run separatly to check the result of each part before pipelining.
* For Advanced Lane Finding, I use the following steps.
1) Generate Parameters for Camera Calibration and Souce/Destination points for perspective transform
2) and Build pipleline for the order : 
Undistort-> Perspective Transform-> Image processing -> detect lane/curvature/location -> warp back(Perspective Transform)-> Joint with original image/video.
  

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###1.Camera Calibration
The code for this step is contained in `cameracalib.py`.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]
![alt text][image2]

and I generated undistored images from folder 'test_images' to use perspective transformation.

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image3]

####2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.
**I performed perspective transform before image processing(color transform, gradient)**

The code for my perspective transform includes a function called `warper()`, which appears in lines 30 through 39 in the file `warper.py`.  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points which I measured manually with other tool in the following manner:

```
 src = np.float32([[316,650],[999,650],[537,490],[751,490]])
 dst = np.float32([[316,650],[999,650],[316,490],[999,490]])


```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 316, 650      | 316, 650      | 
| 999, 650      | 999, 650      |
| 537, 490      | 316, 490      |
| 751, 490      | 999, 490      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

####3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 16 through 72 in `preprocess.py`). 
In the code, I use H channel from HLS Color space to reduce effects from shades and L channel for finding lane. To find more explicit lane image, I handle some threshold for each channel.
Here's an example of my output for this step.

![alt text][image5]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kind of like this:

![alt text][image6]
I did this in lines 6 through 206 in 'find_lanes.py'

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 82 through 100 and 156 through 174 in my code in `find_lanes.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 43 through 48 in my code in `project4.py` in the function `pipeline()`.  Here is an example of my result on a test image:

![alt text][image7]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_images/res_project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My solution is working well with only for the project video. not for challenge videos. I think I need to change my logic with considering size of lane to ignore other line like due to different color on paved and need to consider slopes which can make some distortion with parameters in my perspective transform. and for rapid curves, I should think other way to find path from pictures.

