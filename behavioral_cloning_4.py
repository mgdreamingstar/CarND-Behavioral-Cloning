'''
won't flip left & right camera's images.
'''

import csv
import cv2
import numpy as np
import os

os.chdir('./data')

lines = []
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        steering_center = float(line[3])

        # create adjusted steering measurements for the side camera images

        # steering_left = steering_center + correction
        # steering_right = steering_center - correction

        # # read in images from center, left and right cameras
        # directory
        # img_center = cv2.imread(line[0])
        # img_left = cv2.imread(line[1])
        # img_right = cv2.imread(line[2])

        # # add images and angles to data set
        # car_images.extend(img_center, img_left, img_right)
        # steering_angles.extend(steering_center, steering_left, steering_right)

        lines.append(line)

images = []
measurements = []
correction = 0.2

for line in lines:
    image = cv2.imread(line[0]) # center camera
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)

for line in lines:
    for i in range(1,3):
        image = cv2.imread(line[i].lstrip()) # center camera
        augmented_images.append(image)

        # import pdb
        # pdb.set_trace()

        if i == 1:
            measurement = float(line[3]) + correction
        elif i == 2:
            measurement = float(line[3]) - correction

        augmented_measurements.append(measurement)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

# import pdb
# pdb.set_trace()

# model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D


model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))




model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3)

os.chdir('..')
model.save('model_4.h5')
exit()
