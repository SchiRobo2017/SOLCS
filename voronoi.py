# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 16:55:39 2020

@author: Nakata Koya
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.spatial import voronoi_plot_2d, Voronoi

plt.style.use('ggplot')
point_arr = np.array([
    [10, 10],
    [50, 30],
    [60, 40],
    [30, 80],
    [20, 40],
    [80, 40],
])

point_arr.shape

_ = plt.scatter(x=point_arr[:, 0], y=point_arr[:, 1])

vor = Voronoi(points=point_arr)
_ = voronoi_plot_2d(vor=vor)


