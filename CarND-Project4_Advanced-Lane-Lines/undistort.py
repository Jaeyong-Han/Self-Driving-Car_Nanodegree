#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 15:00:35 2017

@author: jerome
"""
import pickle
import cv2
import os, sys, getopt
from moviepy.editor import VideoFileClip


def load_calib_param() :
    dist_pickle = pickle.load( open( "camera_calibration.p", "rb" ) )
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]
    
    return mtx,dist

def undistort(img,mtx,dist) :
    return cv2.undistort(img, mtx, dist, None, mtx)

def main(argv):
    ifile = ''
    ofile = ''
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print ('undistort.py -i <Source file> -o <Destination file>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('undistort.py -i <Source file> -o <Destination file>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            ifile = arg
        elif opt in ("-o", "--ofile"):
            ofile = arg
    if ifile == '' or ifile == '' :
        print ('undistort.py -t <type> -i <Source file> -o <Destination file>')
        sys.exit()
    
    mtx,dist = load_calib_param()

    def video_undistort(clip, mtx, dist):
        def wrapper(image):
            return undistort(image,mtx,dist)
        return clip.fl_image(wrapper)

    if  ifile.split(".")[-1] == "mp4":
        white_output = os.path.abspath(os.path.curdir)+ofile
        clip = VideoFileClip(os.path.abspath(os.path.curdir)+ifile)
        white_clip = clip.fx(video_undistort, mtx, dist) #NOTE: this function expects color images!!
        white_clip.write_videofile(white_output, audio=False)
    else :
        img = cv2.imread(os.path.abspath(os.path.curdir)+ifile)
        cv2.imwrite(os.path.abspath(os.path.curdir)+ofile,undistort(img,mtx,dist))

if __name__ == "__main__":
   main(sys.argv[1:])

    