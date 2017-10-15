# -*- coding:utf-8 -*-

import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from model import Model
from data_sample import gaussian_mixture, gaussian_mixture_B
from util import plot_scatter


if __name__ == u'__main__':

    # parameter
    epoch_num = 200
     
    z_dim = 256    
    batch_size = 256
    
    # make model
    print('-- make model --')
    model = Model(z_dim, batch_size, 3, 0.1)
    model.set_model()


    # training
    print('-- begin training --')
    num_one_epoch = 50
    
    with tf.Session() as sess:
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        sess.run(init)

        for epoch in range(epoch_num):
            plotting = []
            for i in range(50):
                batch_z = np.random.normal(0, 1, [batch_size, z_dim]).astype(np.float32)            
                tmp = model.generate(sess, batch_z)
                plotting.extend([list(_) for _ in tmp])
            plot_scatter(np.asarray(plotting), 'result', epoch)
            
            print('** epoch {} begin **'.format(epoch))
            g_obj = 0.0
            d_obj = 0.0
            for step in range(num_one_epoch):
                
                for i in range(2):
                    batch_z = np.random.normal(0, 1, [batch_size, z_dim]).astype(np.float32)
                    #batch_inputs = gaussian_mixture(batch_size)
                    batch_inputs = gaussian_mixture_B(batch_size)
                    model.training_disc(sess, batch_z, batch_inputs)                
                batch_z = np.random.normal(0, 1, [batch_size, z_dim]).astype(np.float32)
                #batch_inputs = gaussian_mixture(batch_size)
                batch_inputs = gaussian_mixture_B(batch_size)
                d_obj += model.training_disc(sess, batch_z, batch_inputs)
                g_obj += model.training_gen(sess,  batch_z)

            print('epoch:{}, d_obj = {}, g_obj = {}'.format(epoch,
                                                            d_obj/num_one_epoch,
                                                            g_obj/num_one_epoch))
            saver.save(sess, './model.dump')
