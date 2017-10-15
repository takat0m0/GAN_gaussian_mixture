#! -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
from .variable_util import get_const_variable, get_rand_variable, get_dim

def linear(name, inputs, out_dim, stddev = None):
    in_dim = get_dim(inputs)
    if stddev == None:
        stddev = 1/np.sqrt(in_dim)
    
    w = get_rand_variable(name, [in_dim, out_dim], stddev)
    b = get_const_variable(name, [out_dim], 0.0)
    return tf.matmul(inputs, w) + b

