# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 16:49:35 2020

@author: Nakata Koya
"""

import SOLCSTestData as test
import pyclustering
from pyclustering.cluster import xmeans
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math

N = int(math.sqrt(test.nodes.shape[0]))

data = list(test.dict_map_outliner_excluded.values())
labels = list(test.dict_map_outliner_excluded.keys())

xy = np.unravel_index(labels, (N, N))

fig = plt.figure()

ax = fig.add_subplot(1,1,1)

#ax.scatter(x.ravel(), y.ravel(), s=50, c=c, marker="s")
ax.scatter(xy[0], xy[1], marker="s")

ax.set_title('first scatter plot')
ax.set_xlabel('x')
ax.set_ylabel('y')

#plt.scatter(x=X[:, 0], y=X[:, 1])

data2 = list(zip(xy[0], xy[1]))

initializer = xmeans.kmeans_plusplus_initializer(data=data2, amount_centers=8)

initial_centers = initializer.initialize()
xm = xmeans.xmeans(data=data2, initial_centers=initial_centers)
xm.process()

clusters = xm.get_clusters()

xy = np.unravel_index(labels, (N, N))

pyclustering.utils.draw_clusters(data=data2, clusters=clusters)

for i in range(8):
    print("cluster:", i)
    print(np.array(data)[clusters[i]])