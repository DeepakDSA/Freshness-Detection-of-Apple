#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 01:17:39 2024

@author: deepak
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define dataset directories
train_data_dir = '/Users/deepak/Desktop/APSLab/Fruit And Vegetable Diseases Dataset/'
test_data_dir = '/Users/deepak/Desktop/APSLab/Fruit And Vegetable Diseases Dataset/'

# Data augmentation and normalization
train_datagen = ImageDataGenerator(rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Load training and test datasets
train_set = train_datagen.flow_from_directory(train_data_dir, target_size=(128, 128), batch_size=32, class_mode='binary')
test_set = test_datagen.flow_from_directory(test_data_dir, target_size=(128, 128), batch_size=32, class_mode='binary')

# Build the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_set, steps_per_epoch=len(train_set), epochs=10, validation_data=test_set, validation_steps=len(test_set))

# Save the trained model
model.save('vegetable_freshness_model1.h5')