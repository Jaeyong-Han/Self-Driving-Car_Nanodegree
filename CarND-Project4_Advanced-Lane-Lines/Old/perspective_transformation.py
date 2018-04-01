#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 20:04:30 2017

@author: jerome
"""
import numpy as np
import cv2
import glob
import pickle

img = cv2.imread('test_images/straight_lines2.jpg')
img_size = (img.shape[1], img.shape[0])
src_corner = np.float32([[309,650],[1001,650],[540,490],[748,490]])
#src_corner = np.float32([[315,650],[1002,650],[538,490],[751,490]])
dst_corner = np.float32([[309,650],[1001,650],[309,490],[1001,490]])

M = cv2.getPerspectiveTransform(src_corner,dst_corner)

warped = cv2.warpPerspective(img,M,img_size,flags=cv2.INTER_LINEAR)

cv2.imwrite('test_images/perf_straight_lines2.jpg',warped)
# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
#dist_pickle = {}
#dist_pickle["mtx"] = mtx
#dist_pickle["dist"] = dist
#pickle.dump( dist_pickle, open( "perspective_transform.p", "wb" ) )