#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 16:18:24 2017

@author: adnen
"""
# %% Import libraries and dataset
# --------- Import libraries

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
y_ = tf.placeholder(tf.float32, shape=[None, class_output])
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

# %% Define the fully connected layer1
fc_matrix = tf.reshape(Maxpl2, [-1, 7*7*32])
Wfc1 = tf.Variable(tf.truncated_normal([7*7*32, 512], stddev=0.1))
bfc1 = tf.Variable(tf.constant(0.1, shape=[512]))
fcl1 = tf.matmul(fc_matrix, Wfc1) + bfc1
h_fl1 = tf.nn.relu(fcl1)
# %% Applying dropout
keep_prob = tf.placeholder(tf.float32)
fcl_drop1 = tf.nn.dropout(h_fl1, keep_prob)
# %% Define the fully connected layer2
Wfc2 = tf.Variable(tf.truncated_normal([512, 10], stddev=0.1))
bfc2 = tf.Variable(tf.constant(0.1, shape=[10]))
fcl2 = tf.matmul(fcl_drop1, Wfc2) + bfc2
cnn = tf.nn.softmax(fcl2)

# %% Compute the error
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(cnn),
                                              reduction_indices=[1]))
# Define the optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(cnn)

# Correct prediction
correct_prediction = tf.equal(tf.argmax(cnn, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# %%Run the code
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(1100):
        batch = mnist.train.next_batch(50)
        if i % 50 == 0:
            train_accuracy = accuracy.eval(feed_dict={X: batch[0],
                                                      y_: batch[1],
                                                      keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, float(train_accuracy)))
        train_step.run(feed_dict={X: batch[0], y_: batch[1], keep_prob: 0.5})
        print("test accuracy %g"%accuracy.eval(feed_dict={X: mnist.test.images,
                                                          y_: mnist.test.labels,
                                                          keep_prob: 1.0}))
