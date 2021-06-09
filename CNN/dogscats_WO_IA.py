# Dogs vs Cats Image Classification Without Image Augmentation

# In this database the Images are not of the same size 
# The images are also colored, need to get grayscale--- height X width X depth(RGB channels)
# That means we have to use flatten and resize the images to the same dimensions

from __future__ import absolute_import, division, print_function

# Import Tensorflow
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # Reads data from disk

# Helper Libraries
import os
import math
import numpy as np
import matplotlib.pyplot as plt

import logging

from tensorflow.python.keras.backend import sparse_categorical_crossentropy
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# Data Loading
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
zip_dir = tf.keras.utils.get_file('cats_and_dogs_filterted.zip', origin=_URL, extract=True)
zip_dir_base = os.path.dirname(zip_dir)

# Set up the file path and sets
base_dir = os.path.join(os.path.dirname(zip_dir), 'cats_and_dogs_filtered')
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats') # Dir with train cat imgs
train_dogs_dir = os.path.join(train_dir, 'dogs') # Dir with train dog imgs
validation_cats_dir = os.path.join(validation_dir, 'cats') # Dir with validation cat pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs') # Dir with validation dog pictures

# Check how many images (cats and dogs) we got in train and validation dir
num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))

num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))

total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val

print("Total cats images are: ", num_cats_tr) # 1000
print("Total dogs images are: ", num_dogs_tr) # 1000

print("Total Validation cats images are: ", num_cats_val) # 500
print("Total Validation dogs images are: ", num_dogs_val) # 500
print("--")

print("Total TRAINING images are: ", total_train) # 2000
print("Total VALIDATION images are: ", total_val) # 1000

# Set up variables that will be used on preprocesing

BATCH_SIZE = 100 # Number of training examples to process before updating model variables
IMG_SHAPE = 150 # Training data consists of 150X150 pixel images


# Data Preparation 
# #(remember: read img, decode and convert to grid as per RGB, float tensor, rescale to [0,1])

train_image_generator = ImageDataGenerator(rescale = 1. / 255) # Generator for training data
validation_image_generator = ImageDataGenerator(rescale = 1. / 255) # Generator for validation data

# Load images and apply rescaling

train_data_gen = train_image_generator.flow_from_directory(batch_size = BATCH_SIZE,
                                                         directory = train_dir,
                                                         shuffle = True,
                                                         target_size = (IMG_SHAPE, IMG_SHAPE), #(150,150)
                                                         class_mode = 'binary')

val_data_gen = validation_image_generator.flow_from_directory(batch_size = BATCH_SIZE,
                                                         directory = validation_dir,
                                                         shuffle = False,
                                                         target_size = (IMG_SHAPE, IMG_SHAPE), #(150,150)
                                                         class_mode = 'binary')

# Visualizing Training Images

sample_training_images, _ = next(train_data_gen) # Next returns a Batch from dataset, Tuple
# We disregard the labels we only look the images

# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
    
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5 , figsize = (20,20))
    axes = axes.flatten()
    for img, ax in zip (images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()

plotImages(sample_training_images[:5]) # Plot img 0-4

# Build the model
# FOUR Convolution blocks, with one Max Pool layer each

model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), padding = 'same', activation = tf.nn.relu, input_shape = (150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2), 

        tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu'), # can be written as such 
        tf.keras.layers.MaxPooling2D(2, 2), 

        tf.keras.layers.Conv2D(128, (3, 3), activation = 'relu'),
        tf.keras.layers.MaxPooling2D(2, 2), 

        tf.keras.layers.Conv2D(128, (3, 3), activation = 'relu'), # can be written as such 
        tf.keras.layers.MaxPooling2D(2, 2), 

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation = tf.nn.relu), #Layer with fully connected 512 units
        tf.keras.layers.Dense(2)  # Softmax cause we want probabilities, activation = tf.nn.softmax not needed
])

# 2 - Compile the model

model.compile(optimizer = 'adam',
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), # can be written like so
              metrics = ['accuracy']) # images that are correctly classified

model.summary()

# Train the model

EPOCHS = 15
history = model.fit_generator(
            train_data_gen,
            steps_per_epoch = int(np.ceil(total_train / float(BATCH_SIZE))),
            epochs = EPOCHS,
            validation_data = val_data_gen,
            validation_steps = int(np.ceil(total_val / float(BATCH_SIZE)))
)

# Visualizing results after training the network

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize = (8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label = 'Training Accuracy')
plt.plot(epochs_range, val_acc, label = 'Validation Accuracy')
plt.legend(loc = 'lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label = 'Training Loss')
plt.plot(epochs_range, val_loss, label = 'Validation Loss')
plt.legend(loc = 'upper right')
plt.title('Training and Validation Loss')
plt.savefig('./foo.png')
plt.show()
