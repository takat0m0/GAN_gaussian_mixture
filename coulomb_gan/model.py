# -*- coding:utf-8 -*-

import os
import sys

import numpy as np
import tensorflow as tf

from generator import Generator
from discriminator import Discriminator

from tf_util import flatten
from potential_util import calculate_potential, get_potentials

class Model(object):
    def __init__(self, z_dim, batch_size, coulomb_dim, coulomb_epsilon):

        self.z_dim = z_dim
        self.true_input_dim = 2
        self.lr = 0.0005
        self.batch_size = batch_size
        
        self.coulomb_dim = coulomb_dim
        self.coulomb_epsilon = coulomb_epsilon
        
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
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim])
        self.true_input= tf.placeholder(tf.float32, [self.batch_size, self.true_input_dim])
        
        # -- generator -----------------
        gen_out = self.gen.set_model(self.z, True, False)
        g_logits = self.disc.set_model(gen_out, True, False)

        
        # -- discriminator --------
        d_logits = self.disc.set_model(self.true_input, True, True)

        # -- for losses -----------
        g_logits = tf.reshape(g_logits, [-1])
        d_logits = tf.reshape(d_logits, [-1])        

        potential_gen, potential_input = get_potentials(gen_out, self.true_input,
                                                        self.coulomb_dim, self.coulomb_epsilon)

        loss_d_gen = tf.losses.mean_squared_error(potential_gen, g_logits)
        loss_d_input = tf.losses.mean_squared_error(potential_input, d_logits)
        self.g_obj = tf.reduce_mean(g_logits)
        self.d_obj = loss_d_gen + loss_d_input

        # -- for train op ----------
        self.train_gen  = tf.train.AdamOptimizer(self.lr, beta1 = 0.5).minimize(self.g_obj, var_list = self.gen.get_variables())        
        self.train_disc = tf.train.AdamOptimizer(self.lr, beta1 = 0.5).minimize(self.d_obj, var_list = self.disc.get_variables())
        
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
        return d_obj
    
    def generate(self, sess, z):
        ret_ = sess.run(self.gen_figs,
                       feed_dict = {self.z: z})
        return ret_

if __name__ == u'__main__':
    model = Model(30, 100, 3, 0.1)
    model.set_model()
    
