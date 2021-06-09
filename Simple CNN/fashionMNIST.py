# Set up an NN to recognize clothing
# Use 85% of MNIST data to train and 15% to test 
# We will also used ReLU 

from __future__ import absolute_import, division, print_function

# Import Tensorflow
import tensorflow as tf
import tensorflow_datasets as tfds
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) # updated

# Helper Libraries
import math
import numpy as np
import matplotlib.pyplot as plt

# Improve progress bar display
import tqdm
import tqdm.auto
tqdm.tqdm = tqdm.auto.tqdm

#print(tf.__version__)

# Load dataset and metadata
dataset, metadata = tfds.load('fashion_mnist', as_supervised = True, with_info = True)
train_dataset  = dataset['train']
test_dataset = dataset['test']

class_names = metadata.features['label'].names
print("Class names: {}" .format(class_names))

# Explore Data
num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples
print("Number of training examples: {}" .format(num_train_examples)) # 60000
print("Number of test examples: {}" .format(num_test_examples)) # 10000

# Preprocess the data

# Image has pixels with values [0, 255] ---- NORMALIZATION
def normalize(images, labels):
    images = tf.cast(images, tf.float32) # cast it as float
    images /= 255 # Casting, to return a value between 0 and 1
    return images, labels

# Map function applies normalize function to each element in the followin sets
train_dataset  = train_dataset.map(normalize)
test_dataset  = test_dataset.map(normalize)

"""
# Plot the first image of test_dataset
for image, label in test_dataset.take(1):
    break
image = image.numpy().reshape((28, 28))
# Plot the image 
plt.figure()
plt.imshow(image, cmap = plt.cm.binary)
plt.colorbar
plt.grid(False)
plt.show()

# Diplay the first 25 imgaes from Training Set and display the class
plt.figure(figsize=(10, 10))
i = 0
for (image, label) in test_dataset.take(25):
    image = image.numpy().reshape((28, 28))
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image, cmap = plt.cm.binary)
    plt.xlabel(class_names[label])
    i +=1
plt.show()
"""

# Build the model

# 1 - Set up Layers

model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape = (28, 28, 1)), #  Image from 2d array of 28X28 to 1D of 784
        tf.keras.layers.Dense(128, activation = tf.nn.relu), # Densely connected hidden Layer of 128 Neurons
        tf.keras.layers.Dense(10, activation = tf.nn.softmax) # 10-node softmax layer, each node is a clothing class
])# Input-hidden-output

# 2 - Compile the model
model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy']) # images that are correctly classified

# 3 - Train the model

BATCH_SIZE = 32
train_dataset = train_dataset.repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)
model.fit(train_dataset, epochs = 5, steps_per_epoch = math.ceil(num_train_examples / BATCH_SIZE))
# Notice improving accuracy that reaches 0,89

# 4 - Evaluate Accuracy
test_loss, test_accuracy = model.evaluate(test_dataset, steps = math.ceil(num_test_examples / 32))
print("Accuracy on test dataset: ", test_accuracy) # 0,87

# 5 - Predictions and Exploration

for test_images, test_labels in test_dataset.take(1):
    test_images = test_images.numpy()
    test_labels = test_labels.numpy()
    predictions = model.predict(test_images)
    print(predictions.shape) # (32,10) 32 answers 10 classes
print(predictions[0]) # For 1st image
print(np.argmax(predictions[0])) # Class 4 to take the largest prediction
test_labels[0]

# Plot the results on full 10 channel set
def plot_image(i, predictions_array, true_labels, images):
    predictions_array, true_label, img = predictions_array[i], true_labels[i], images[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img[..., 0], cmap = plt.cm.binary)
    

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                        100 * np.max(predictions_array),
                                        class_names[true_label]),
                                        color = color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color = "#777777")
    plt.ylim([0,1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')
    

# Check our  for a certain pic
"""
i = 12 # a Pullover
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1, 2, 1)
plot_value_array(i, predictions, test_labels)
"""

# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()

# Grab an image from the test dataset
img = test_images[0]
print(img.shape)
# Add the image to a batch where it's the only member.
img = np.array([img])
print(img.shape)
# now predict
predictions_single = model.predict(img)
print(predictions_single)
plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()
print(np.argmax(predictions_single[0]))