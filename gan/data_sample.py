# -*- coding:utf-8 -*-

import os
import sys
import numpy as np
import math
from util import plot_scatter

data_points_B = np.asarray([
        [-0.5, 1], [0, 1], [0.25, 0.75], [-0.5, 0.5], [0.5, 0.5],
        [-0.5, 0], [0, 0], [0.25, -0.25], [-0.5, -0.5], [0.5, -0.5],
        [0.25, -0.75], [-0.5, -1],[0, -1], [0.25, 0.25], [-0.25, 0]
        ]) * 2

def gaussian_mixture_B(batchsize, std = 0.1):
        mean = data_points_B[np.random.choice(range(len(data_points_B)), batchsize)]
        return np.random.normal(mean, std**2, (batchsize, 2)).astype(np.float32)

        
def gaussian_mixture(batchsize, num_cluster=8, scale=2, std=0.2):
        rand_indices = np.random.randint(0, num_cluster, size=batchsize)
        base_angle = math.pi * 2 / num_cluster
        angle = rand_indices * base_angle - math.pi / 2
        mean = np.zeros((batchsize, 2), dtype=np.float32)
        mean[:, 0] = np.cos(angle) * scale
        mean[:, 1] = np.sin(angle) * scale
        return np.random.normal(mean, std**2, (batchsize, 2)).astype(np.float32)

if __name__ == '__main__':
        plot_scatter(gaussian_mixture_B(1000), dir = '.', color = 'red')        

