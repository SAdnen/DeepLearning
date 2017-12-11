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