# -*- coding:utf-8 -*-

import os
import sys
import tensorflow as tf

class Layers(object):
    def __init__(self, name_scopes):
        self.name_scopes = name_scopes
        
    def _check_name_scope(self, var_name):
        ret = False
        for ns in self.name_scopes:
            ret += True if ns in var_name else False
        return ret
    
    def get_variables(self):
        t_var = tf.trainable_variables()
        ret = []
        for var in t_var:
            if self._check_name_scope(var.name):
                ret.append(var)
        return ret
    def set_model(self):
        pass
