# -*- coding:utf-8 -*-

import os
import sys

import numpy as np
import tensorflow as tf

from generator import Generator
from discriminator import Discriminator

from tf_util import flatten

class Model(object):
    def __init__(self, z_dim):

        self.z_dim = z_dim
        self.true_input_dim = 2
        self.lr = 0.0005
        
        # generator config
        gen_layer = [z_dim, 128, 128, self.true_input_dim]

        #discriminato config
        disc_layer = [self.true_input_dim, 128, 128, 1]

        # -- generator -----
        self.gen = Generator([u'gen_deconv'], gen_layer)

        # -- discriminator --
        self.disc = Discriminator([u'disc_conv'], disc_layer)

    def set_model(self):

        # -- define place holder -------
        self.z = tf.placeholder(tf.float32, [None, self.z_dim])
        self.true_input= tf.placeholder(tf.float32, [None, self.true_input_dim])
        
        # -- generator -----------------
        gen_out = self.gen.set_model(self.z, True, False)
        g_logits = self.disc.set_model(gen_out, True, False)

        self.g_obj =  -tf.reduce_mean(
            tf.reduce_sum(g_logits, 1)
        )

        self.train_gen  = tf.train.RMSPropOptimizer(self.lr).minimize(self.g_obj, var_list = self.gen.get_variables())
        
        # -- discriminator --------
        d_logits = self.disc.set_model(self.true_input, True, True)

        d_obj_true = tf.reduce_mean(
            tf.reduce_sum(d_logits, 1)
        )
        d_obj_false = tf.reduce_mean(
            tf.reduce_sum(g_logits, 1)
        )
        
        d_weight_decay = 0
        for i in self.disc.get_variables():
            d_weight_decay += tf.nn.l2_loss(i)

        self.d_obj = -d_obj_true + d_obj_false
        self.train_disc = tf.train.RMSPropOptimizer(self.lr).minimize(self.d_obj, var_list = self.disc.get_variables())

        # -- for clipping ----------------
        c = 0.01
        self.clip_D = [_.assign(tf.clip_by_value(_, -c, c)) for _ in self.disc.get_variables()]
        
        # -- for figure generation -------
        self.gen_figs = self.gen.set_model(self.z, False, True)
        
    def training_gen(self, sess, z_list):
        _, g_obj = sess.run([self.train_gen, self.g_obj],
                            feed_dict = {self.z: z_list})
        return g_obj
        
    def training_disc(self, sess, z_list, figs):
        _, d_obj = sess.run([self.train_disc, self.d_obj],
                            feed_dict = {
                                         self.z: z_list,
                                         self.true_input:figs})
        sess.run([self.clip_D])
        return d_obj
    
    def generate(self, sess, z):
        ret_ = sess.run(self.gen_figs,
                       feed_dict = {self.z: z})
        return ret_

if __name__ == u'__main__':
    model = Model(z_dim = 30)
    model.set_model()
    
