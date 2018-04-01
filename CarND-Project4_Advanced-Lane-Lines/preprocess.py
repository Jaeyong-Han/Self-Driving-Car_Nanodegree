#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 15:00:35 2017

@author: jerome
"""
import cv2
import numpy as np
import os, sys, getopt
from moviepy.editor import VideoFileClip

def wrap_process_image(img) :
    return cv2.cvtColor(process_image(img)*255,cv2.COLOR_GRAY2RGB) 

def process_image(img) :
    hlsImg = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    h_ch = hlsImg[:,:,0]
    l_ch = hlsImg[:,:,1]
    s_ch = hlsImg[:,:,2]
    h_ch = 255-h_ch
    ls_ch = np.uint8(np.dot(hlsImg,[0,0.5,0.5]))
    
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
    
     # Threshold color channel
    ls_thresh_min = 100
    ls_thresh_max = 200
    ls_binary = np.zeros_like(h_ch)
    ls_binary[(ls_ch >= ls_thresh_min) & (ls_ch <= ls_thresh_max)] = 1
    
    
      
    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    
    #color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
    
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[((s_binary == 1) & (h_binary == 1)) | (sxbinary == 1)] = 1
    #combined_binary[(ls_binary == 1) & (sxbinary == 1)] = 1
    #combined_binary = np.hstack((scaled_sobel,s_ch,h_ch))
    #fnl_img = cv2.cvtColor(find_lanes.find_lane(combined_binary),cv2.COLOR_GRAY2RGB) 

    return combined_binary
    
def main(argv):
    ifile = ''
    ofile = ''
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print ('preprocess.py -i <Source file> -o <Destination file>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('preprocess.py -i <Source file> -o <Destination file>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            ifile = arg
        elif opt in ("-o", "--ofile"):
            ofile = arg
    if ifile == '' or ifile == '' :
        print ('preprocess.py -i <Source file> -o <Destination file>')
        sys.exit()
    if  ifile.split(".")[-1] == "mp4":
        white_output = os.path.abspath(os.path.curdir)+ofile
        clip1 = VideoFileClip(os.path.abspath(os.path.curdir)+ifile)
        white_clip = clip1.fl_image(wrap_process_image) #NOTE: this function expects color images!!
        white_clip.write_videofile(white_output, audio=False)
    else :
        img = cv2.imread(os.path.abspath(os.path.curdir)+ifile)
        cv2.imwrite(os.path.abspath(os.path.curdir)+ofile,process_image(img)*255)

if __name__ == "__main__":
   main(sys.argv[1:])
