#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 20:04:30 2017
####################################

Perspective Transformation

####################################
@author: jerome
"""
import numpy as np
import cv2
import pickle
import os, sys, getopt
from moviepy.editor import VideoFileClip

def load_corner():
    dist_pickle = pickle.load( open( "corner.p", "rb" ) )
    src = dist_pickle["src"]
    dst = dist_pickle["dst"]
    return src,dst

def save_corner(src, dst):
    dist_pickle = {}
    dist_pickle["src"] = src
    dist_pickle["dst"] = dst
    pickle.dump( dist_pickle, open( "corner.p", "wb" ) )
  
def warper(img,img_size, src, dst):    
    
    M = cv2.getPerspectiveTransform(src,dst)
    
    return cv2.warpPerspective(img,M,img_size,flags=cv2.INTER_LINEAR)

def video_warper(clip,src,dst):
    def wrapper(image):          
        return warper(image,(image.shape[1], image.shape[0]),src,dst)
    return clip.fl_image(wrapper)

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
        print ('undistort.py -i <Source file> -o <Destination file>')
        sys.exit()   
    
    src = np.float32([[316,650],[999,650],[537,490],[751,490]])
    #src = np.float32([[315,650],[1002,650],[538,490],[751,490]])
    dst = np.float32([[316,650],[999,650],[316,490],[999,490]])
    
    if  ifile.split(".")[-1] == "mp4":
        white_output = os.path.abspath(os.path.curdir)+ofile
        clip = VideoFileClip(os.path.abspath(os.path.curdir)+ifile)
        white_clip = clip.fx(video_warper, src, dst) #NOTE: this function expects color images!!
        white_clip.write_videofile(white_output, audio=False)
    else :
        img = cv2.imread(os.path.abspath(os.path.curdir)+ifile)
        img_size = (img.shape[1], img.shape[0])
        cv2.imwrite(os.path.abspath(os.path.curdir)+ofile,warper(img,img_size,src,dst))

if __name__ == "__main__":
   main(sys.argv[1:])

