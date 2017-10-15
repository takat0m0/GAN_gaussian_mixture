#! -*- coding:utf-8 -*-

import os
import sys
import numpy as np
import tensorflow as tf

from tf_util import conv, linear, Layers, batch_norm, lrelu, flatten

class Discriminator(Layers):
    def __init__(self, name_scopes, layer_channels):
        assert(len(name_scopes) == 1)
        super().__init__(name_scopes)
        self.layer_channels = layer_channels

    def set_model(self, inputs,  is_training = True, reuse = False):
        
        h  = inputs
        
        # fully connect
        with tf.variable_scope(self.name_scopes[0], reuse = reuse):
            last_id = len(self.layer_channels) - 1
            for i, s in enumerate(self.layer_channels):
                if i != last_id:
                    lin = linear(i, h, s)
                else:
                    lin = linear(i, h, s, 0.0)
                h = lrelu(lin)
        return lin

if __name__ == u'__main__':
    dis = Discriminator([u'disc_conv'], [2, 64, 128, 256, 1])
                        
    imgs = tf.placeholder(tf.float32, [None, 2])
    h = dis.set_model(imgs)
    h = dis.set_model(imgs, True, True)    
    print(h)
