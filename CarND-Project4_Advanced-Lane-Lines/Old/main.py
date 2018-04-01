#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 15:00:35 2017

@author: jerome
"""
from PIL import Image
import pickle
import cv2
import numpy as np
from moviepy.editor import VideoFileClip

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)

# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
dist_pickle = pickle.load( open( "camera_calibration.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

def process_image(img) :
    # MODIFY THIS FUNCTION TO GENERATE OUTPUT 
    # THAT LOOKS LIKE THE IMAGE ABOVE
    img = cv2.undistort(img, mtx, dist, None, mtx)
    hlsImg = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    h_ch = hlsImg[:,:,0]
    s_ch = hlsImg[:,:,2]
    h_ch = 255-h_ch
    
    kernel_size = 3
        
    gryImg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gryImg = cv2.GaussianBlur(gryImg, (kernel_size, kernel_size), 0)
    
    # Sobel x
    sobelx = cv2.Sobel(gryImg, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    thresh_min = 20
    thresh_max = 100
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    
    # Threshold color channel
    s_thresh_min = 70
    s_thresh_max = 255
    s_binary = np.zeros_like(s_ch)
    s_binary[(s_ch >= s_thresh_min) & (s_ch <= s_thresh_max)] = 1
    
    # Threshold color channel
    h_thresh_min = 170
    h_thresh_max = 255
    h_binary = np.zeros_like(h_ch)
    h_binary[(h_ch >= h_thresh_min) & (h_ch <= h_thresh_max)] = 1
    
    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    
    #color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
    
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[((s_binary == 1) & (h_binary == 1)) | (sxbinary == 1)] = 1
    
    fnl_img = cv2.cvtColor(combined_binary*255,cv2.COLOR_GRAY2RGB) 
    
    img_size = (fnl_img.shape[1], fnl_img.shape[0])
    src_corner = np.float32([[326,650],[990,650],[545,490],[747,490]])
    #src_corner = np.float32([[315,650],[1002,650],[538,490],[751,490]])
    dst_corner = np.float32([[326,650],[990,650],[326,490],[990,490]])
    
    M = cv2.getPerspectiveTransform(src_corner,dst_corner)
    
    warped = cv2.warpPerspective(fnl_img,M,img_size,flags=cv2.INTER_LINEAR)

    return warped
    #return  cv2.cvtColor((combined_binary*255),cv2.COLOR_GRAY2RGB) 
    
    ###########
    # Plotting thresholded images
    #return cv2.cvtColor((combined_binary*255),cv2.COLOR_GRAY2RGB) 
    #return cv2.cvtColor(h_binary,cv2.COLOR_GRAY2RGB) 

white_output = 'project_video_cv.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

