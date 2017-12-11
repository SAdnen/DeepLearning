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

# %% Define Placeholders and Variables

X = tf.placeholder(tf.float32, shape=(1, 28, 28, 1), name="Input_Image")
W1 = tf.Variable(tf.truncated_normal((5, 5, 1, 1), stddev=0.1))
b1 = tf.Variable(tf.zeros(1, 1))

# %% Define first confolutional layer

Conv1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME') + b1
Conv1out = tf.nn.relu(Conv1)
Maxpl1 = tf.nn.max_pool(Conv1out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                        padding="SAME")
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    result = sess.run(Maxpl1, feed_dict={X: x})




















