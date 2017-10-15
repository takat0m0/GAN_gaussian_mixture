#! -*- coding:utf-8 -*-

import os
import sys
import cv2
import numpy as np
import tensorflow as tf


'''
Reference: https://github.com/bioinf-jku/coulomb_gan
'''

def calculate_squared_distances(a, b):

    na = tf.shape(a)[0]
    nb = tf.shape(b)[0]
    nas, nbs = a.get_shape().as_list(), b.get_shape().as_list()
    a = tf.reshape(a, [na, 1, -1])
    b = tf.reshape(b, [1, nb, -1])
    a.set_shape([nas[0], 1, np.prod(nas[1:])])
    b.set_shape([1, nbs[0], np.prod(nbs[1:])])
    a = tf.tile(a, [1, nb, 1])
    b = tf.tile(b, [na, 1, 1])
    d = a-b
    return tf.reduce_sum(tf.square(d), axis=2)


def plummer_kernel(a, b, dimension, epsilon):
    r = calculate_squared_distances(a, b)
    r += epsilon*epsilon
    f1 = dimension-2
    return tf.pow(r, -f1 / 2)


def get_potentials(x, y, dimension, cur_epsilon):

    x_fixed = tf.stop_gradient(x)
    y_fixed = tf.stop_gradient(y)
    nx = tf.cast(tf.shape(x)[0], x.dtype)
    ny = tf.cast(tf.shape(y)[0], y.dtype)
    pk_xx = plummer_kernel(x_fixed, x, dimension, cur_epsilon)
    pk_yx = plummer_kernel(y, x, dimension, cur_epsilon)
    pk_yy = plummer_kernel(y_fixed, y, dimension, cur_epsilon)
    pk_xx = tf.matrix_set_diag(pk_xx, tf.ones(shape=x.get_shape()[0], dtype=pk_xx.dtype))
    pk_yy = tf.matrix_set_diag(pk_yy, tf.ones(shape=y.get_shape()[0], dtype=pk_yy.dtype))
    kxx = tf.reduce_sum(pk_xx, axis=0) / (nx)
    kyx = tf.reduce_sum(pk_yx, axis=0) / ny
    kxy = tf.reduce_sum(pk_yx, axis=1) / (nx)
    kyy = tf.reduce_sum(pk_yy, axis=0) / ny
    pot_x = kxx - kyx
    pot_y = kxy - kyy
    pot_x = tf.reshape(pot_x, [-1])
    pot_y = tf.reshape(pot_y, [-1])
    return pot_x, pot_y


def calculate_potential(x, y, a, dimension, epsilon):
    x = tf.stop_gradient(x)
    y = tf.stop_gradient(y)
    nx = tf.cast(tf.shape(x)[0], x.dtype)
    ny = tf.cast(tf.shape(y)[0], y.dtype)
    kxa = plummer_kernel(x, a, dimension, epsilon)
    kxa = tf.reduce_sum(kxa, axis=0) / nx
    kya = plummer_kernel(y, a, dimension, epsilon)
    kya = tf.reduce_sum(kya, axis=0) / ny
    p = kxa - kya
    p = tf.reshape(p, [-1])
    return p
