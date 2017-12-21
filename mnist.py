#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 16:18:24 2017

@author: adnen
"""
# %% Import libraries and dataset
# --------- Import libraries

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# --------- Import Mnist dataset

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
plt.rcParams['figure.figsize'] = (5.0, 5.0)
sampleimage = mnist.test.images[2].reshape([28, 28])
plt.imshow(sampleimage, cmap="gray")
x = sampleimage.reshape([1, 28, 28, 1])
# %% Initial Parameters
width = 28  # width of the image in pixels
height = 28  # height of the image in pixels
flat = width * height  # number of pixels in one image
class_output = 10  # number of possible classifications for the problem

# %% Input and Output
X = tf.placeholder(tf.float32, shape=[None, flat], name="Input_Image")
y = tf.placeholder(tf.int32, shape=[None, class_output])
x_image = tf.reshape(X, shape=[-1, 28, 28, 1])

# %% Define Placeholders and Variables of the first layer
# Convolutional layer composed of 16 filters of 5x5 size
W1 = tf.Variable(tf.truncated_normal([5, 5, 1, 16], stddev=0.1))
b1 = tf.Variable(tf.constant(0.1, shape=[16]))  # 16 biases
# %% Define first convolutional layer
Conv1 = tf.nn.conv2d(x_image, W1, strides=[1, 1, 1, 1], padding='SAME') + b1
Conv1out = tf.nn.relu(Conv1)
Maxpl1 = tf.nn.max_pool(Conv1out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                        padding="SAME")
# Shape Maxpl1 = (1, 14, 14, 16)
# %% Define Placeholders and Variables of the second layer
# Convolutional Layer composed of 32 filters of 5x5 size
W2 = tf.Variable(tf.truncated_normal([5, 5, 16, 32], stddev=0.1))
b2 = tf.Variable(tf.constant(0.1, shape=[32]))

# %% Define the second convolutional layer

Conv2 = tf.nn.conv2d(Maxpl1, W2, strides=[1, 1, 1, 1], padding='SAME') + b2
Convout2 = tf.nn.relu(Conv2)
Maxpl2 = tf.nn.max_pool(Convout2,  ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                        padding='SAME')
# Shape Maxpl2 = (1, 7, 7, 32)

# %% Define the fully connected layer

fc_matrix = tf.reshape(Maxpl2,[-1, 7*7, 32] )


init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    result = sess.run(Maxpl1, feed_dict={X: x})




















