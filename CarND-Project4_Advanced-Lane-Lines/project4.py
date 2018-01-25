#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 15:00:35 2017

@author: jerome
"""
import cv2
import numpy as np
import os, sys, getopt
import undistort
import warper
import preprocess
import find_lanes

from moviepy.editor import VideoFileClip

left_fit = ''
right_fit = ''
mask = ''
mtx = ''
dist = ''

def pipeline(img):
    global left_fit, right_fit, mask, mtx, dist
    #src = np.float32([[316,650],[999,650],[537,490],[751,490]])
    #dst = np.float32([[316,650],[999,650],[316,490],[999,490]])
    src = np.float32(
        [[(img.shape[1] / 2) - 55, img.shape[0] / 2 + 100],
        [((img.shape[1] / 6) - 10), img.shape[0]],
        [(img.shape[1] * 5 / 6) + 60, img.shape[0]],
        [(img.shape[1] / 2 + 55), img.shape[0] / 2 + 100]])
    dst = np.float32(
        [[(img.shape[1] / 4), 0],
        [(img.shape[1] / 4), img.shape[0]],
        [(img.shape[1] * 3 / 4), img.shape[0]],
        [(img.shape[1] * 3 / 4), 0]])

    und_img = undistort.undistort(img, mtx, dist)
    war_img = warper.warper(und_img,(und_img.shape[1], und_img.shape[0]),src,dst)
    binary_warped = preprocess.process_image(war_img)
    outimage, left_fit, right_fit, mask,curverad,locx = find_lanes.find_lane(binary_warped,left_fit,right_fit,mask)
    cv2.putText(img,"Radius of Curvature = "+ str(curverad)+"(m)",(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(200,255,155))
    if locx < 0 :
        cv2.putText(img,"Vehicle is "+ '{:.2f}'.format(abs(locx))+"m left of center",(30,60),cv2.FONT_HERSHEY_SIMPLEX,1,(200,255,155))
    else :
        cv2.putText(img,"Vehicle is "+ '{:.2f}'.format(abs(locx))+"m right of center",(30,60),cv2.FONT_HERSHEY_SIMPLEX,1,(200,255,155))
    return cv2.addWeighted(img, 1,warper.warper(outimage,(outimage.shape[1], outimage.shape[0]),dst,src), 0.5, 0)

def main(argv):
    global mtx,dist
    ifile = ''
    ofile = ''
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print ('project4.py -i <Source file> -o <Destination file>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('project4.py -i <Source file> -o <Destination file>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            ifile = arg
        elif opt in ("-o", "--ofile"):
            ofile = arg
    if ifile == '' or ifile == '' :
        print ('project4.py -i <Source file> -o <Destination file>')
        sys.exit()
    mtx,dist = undistort.load_calib_param()
    if  ifile.split(".")[-1] == "mp4":
        white_output = os.path.abspath(os.path.curdir)+ofile
        clip = VideoFileClip(os.path.abspath(os.path.curdir)+ifile)
        white_clip = clip.fl_image(pipeline) #NOTE: this function expects color images!!
        #white_clip = clip.fx(pipeline, mtx, dist)
        white_clip.write_videofile(white_output, audio=False)
    else :
        img = cv2.imread(os.path.abspath(os.path.curdir)+ifile)
        cv2.imwrite(os.path.abspath(os.path.curdir)+ofile,pipeline(img))

if __name__ == "__main__":
   main(sys.argv[1:])
