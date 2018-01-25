#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 07:12:53 2017

@author: jerome Jaeyong Han
"""
import csv
import cv2
import numpy as np
import sklearn
from random import shuffle

samples = []

"""
Load driving logs and adjust angle of steering wheel
"""

norm_range = 3

## Driving logs from track1 and track 2
log_list = ["driving_log_track1_0.csv","driving_log_track1_1.csv",
            "driving_log_track1_2.csv","driving_log_track1_3.csv",
            "driving_log_track2_1.csv","driving_log_track2_2.csv",
            "driving_log_track2_3.csv","driving_log_track2_4.csv"]

## Driving logs from track1
#log_list = ["driving_log_track1_0.csv","driving_log_track1_1.csv",
#            "driving_log_track1_2.csv","driving_log_track1_3.csv"]

## Driving logs from track2
#log_list = ["driving_log_track2_1.csv","driving_log_track2_2.csv",
#            "driving_log_track2_3.csv","driving_log_track2_4.csv"]

##Load csv files
for fname in log_list:
    lines = []
    with open('./data/'+fname) as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        for line in reader:
            lines.append(line)
    
    #Adjust Steering wheel angle(discrete to psuedo-continuous)
    for idx in range(len(lines)-norm_range): 
        line_sum = 0.0
        for line in lines[idx:idx+norm_range]:
            line_sum = line_sum + float(line[3])
        lines[idx][3] = line_sum*1.0/norm_range
    samples += lines[0:(len(lines)-norm_range)] #Trim end laps which don't have enough data for angle adjustment


from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
"""
1) Load image with Generator
2) Add more data using left/right images with corrected angle and flipping image for opposite direction
"""                                
images = []
measurements = []
correction = 0.2

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: 
        shuffle(samples) 
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            # angle correction for left/right images
            for batch_sample in batch_samples:
                for i in range(3):
                    source_path = batch_sample[i]
                    filename = source_path.split('/')[-1]
                    current_path = './data/IMG/' + filename
                    image = cv2.imread(current_path)
                    images.append(image)
                    angle = float(batch_sample[3])
                    if (i==1) :
                        angle += correction
                    if (i==2) :
                        angle -= correction
                    angles.append(angle)
            #flipping image to have opposite direction data
            augmented_images, augmented_angles = [], []
            for image, angle in zip(images, angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                augmented_images.append(cv2.flip(image,1))
                augmented_angles.append(angle*-1.0)
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)
            yield sklearn.utils.shuffle(X_train, y_train)

"""
Modeling with CNN model from NVIDIA paper (End to End Learning for Self-Driving Cars)
"""                        
train_generator = generator(train_samples, batch_size=10)
validation_generator = generator(validation_samples, batch_size=10)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D

#Build NVIDIA model with keras
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.2)) #Add dropout to avoid overfitting
model.add(Dense(50))
model.add(Dropout(0.2)) #Add dropout to avoid overfitting
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam') #Use adam optimizer
model.fit_generator(train_generator, samples_per_epoch= len(train_samples), 
                    validation_data=validation_generator, 
                    nb_val_samples=len(validation_samples), nb_epoch=150)
#Save model. 
model.save('model.h5')
