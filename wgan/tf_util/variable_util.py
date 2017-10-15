# -*- coding:utf-8 -*-

import os
import sys
import numpy as np
import tensorflow as tf

def get_rand_variable(name, shape, stddev, trainable = True):
    return tf.get_variable('weights{}'.format(name), shape,
                           initializer = tf.random_normal_initializer(stddev = stddev),
                           trainable = trainable)

def get_const_variable(name, shape, value, trainable = True):
    return tf.get_variable('biases{}'.format(name), shape,
                           initializer = tf.constant_initializer(value),
                           trainable = trainable)

def get_dim(target):
    dim = 1
    for d in target.get_shape()[1:].as_list():
        dim *= d
    return dim

def flatten(inputs):
    return tf.reshape(inputs, [-1, get_dim(inputs)])
