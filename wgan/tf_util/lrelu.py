#! -*- coding:utf-8 -*-

import tensorflow as tf

def lrelu(x, leak=0.2):
  return tf.maximum(x, leak*x)
