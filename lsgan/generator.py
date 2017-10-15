#! -*- coding:utf-8 -*-

import os
import sys
import numpy as np
import tensorflow as tf

from tf_util import deconv, linear, Layers, batch_norm

class Generator(Layers):
    def __init__(self, name_scopes, layer_chanels):
        assert(len(name_scopes) == 1)
        super().__init__(name_scopes)
        self.layer_chanels = layer_chanels

    def set_model(self, z, is_training = True, reuse = False):

        h = z

        # deconvolution
        with tf.variable_scope(self.name_scopes[0], reuse = reuse):
            for i, out_chan in enumerate(self.layer_chanels[1:]):
                lin = linear(i, h, out_chan)
                h = tf.nn.relu(lin)
        return lin

    
if __name__ == u'__main__':
    g = Generator([u'deconvolution'],
                  [256, 128, 64, 2])
    z = tf.placeholder(tf.float32, [None, 256])
    h = g.set_model(z)
    h = g.set_model(z, True, True)    
    print(h)
