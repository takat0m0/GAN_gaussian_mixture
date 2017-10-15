#! -*- coding:utf-8 -*-

import os
import sys
import cv2
import numpy as np
import pylab

def _resizing(img):
    #return cv2.resize(img, (256, 256))
    return cv2.resize(img, (32, 32))

def _reg(img):
    return img/127.5 - 1.0

def _re_reg(img):
    return (img + 1.0) * 127.5

def get_figs(target_dir):
    ret = []
    for file_name in os.listdir(target_dir):
        target_file = os.path.join(target_dir, file_name)
        img = cv2.imread(target_file, 0)
        ret.append(_reg(_resizing(img)))
    return np.asarray(ret, dtype = np.float32)

def dump_figs(figs, dump_dir):
    for i, fig in enumerate(figs):
        target_file = os.path.join(dump_dir, '{}.jpg'.format(i))
        cv2.imwrite(target_file, _re_reg(fig))

def plot_scatter(data, dir=None, filename="scatter", color="blue"):
	if dir is None:
		raise Exception()
	try:
		os.mkdir(dir)
	except:
		pass
	fig = pylab.gcf()
	fig.set_size_inches(16.0, 16.0)
	pylab.clf()
	pylab.scatter(data[:, 0], data[:, 1], s=20, marker="o", edgecolors="none", color=color)
	pylab.xlim(-4, 4)
	pylab.ylim(-4, 4)
	pylab.savefig("{}/{}.png".format(dir, filename))        
