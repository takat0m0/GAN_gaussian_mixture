#! -*- coding:utf-8 -*-

import os
import sys
import numpy as np
import tensorflow as tf

from .variable_util import get_const_variable

def batch_norm(name, x, decay_rate = 0.99, is_training = True):
    #decay_rate = 0.99
    
    shape = x.get_shape().as_list()
    dim = shape[-1]
    if len(shape) == 2:
        mean, var = tf.nn.moments(x, [0], name = 'moments_bn_{}'.format(name))
    elif len(shape) == 4:
        mean, var = tf.nn.moments(x, [0, 1, 2], name = 'moments_bn_{}'.format(name))

    avg_mean  = get_const_variable('avg_mean_bn_{}'.format(name),
                                   [1, dim], 0.0, False)
    
    avg_var = get_const_variable('avg_var_bn_{}'.format(name),
                                 [1, dim], 1.0, False)
    
    beta  = get_const_variable('beta_bn_{}'.format(name),
                               [1, dim], 0.0)
    gamma = get_const_variable('gamma_bn_{}'.format(name),
                               [1, dim], 1.0)

    if is_training:
        avg_mean_assign_op = tf.assign(avg_mean, decay_rate * avg_mean
                                       + (1 - decay_rate) * mean)
        avg_var_assign_op = tf.assign(avg_var,
                                      decay_rate * avg_var
                                      + (1 - decay_rate) * var)

        with tf.control_dependencies([avg_mean_assign_op, avg_var_assign_op]):
            ret = gamma * (x - mean) / tf.sqrt(1e-6 + var) + beta
    else:
        ret = gamma * (x - avg_mean) / tf.sqrt(1e-6 + avg_var) + beta
        
    return ret

if __name__ == u'__main__':
    x = tf.placeholder(dtype = tf.float32, shape = [None, 10, 10, 3])
    batch_norm(1, x, 0.9, True)
