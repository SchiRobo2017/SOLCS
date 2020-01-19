# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 14:11:18 2020

@author: Nakata Koya
"""

import math
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import SOLCSTestData as test
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

N = int(math.sqrt(test.nodes.shape[0]))

#scatter: faster than heatmap by imshow
"""
#data = np.random.rand(16*21).reshape(21,16)
data = list(test.dict_map_outliner_excluded.values())
labels = list(test.dict_map_outliner_excluded.keys())
df = pd.DataFrame(data)
Z = linkage(df,method="ward",metric="euclidean")

r = dendrogram(Z, p=10, truncate_mode="level", labels=labels)

t = 0.7*max(Z[:,2])
c = fcluster(Z, t=8, criterion="maxclust")

#x, y = np.meshgrid(range(N), range(N))

xy = np.unravel_index(labels, (N, N))

fig = plt.figure()

ax = fig.add_subplot(1,1,1)

#ax.scatter(x.ravel(), y.ravel(), s=50, c=c, marker="s")
ax.scatter(xy[0], xy[1], c=c, marker="s")

ax.set_title('first scatter plot')
ax.set_xlabel('x')
ax.set_ylabel('y')
"""

#heatmap by imshow
#data2 = [ [x**2+y**2 for x in range(5)] for y in range(5)]
data2 = test.map_new_input
df2 = pd.DataFrame(data2)
Z2 = linkage(df2,method="ward",metric="euclidean")

#r2 = dendrogram(Z2)

c2 = fcluster(Z2, t=8, criterion="maxclust")

#plt.imshow(thresh, 'gray', vmin = 0, vmax = 255) 
plt.imshow(c2.reshape(N,N)-1, cmap="gray_r", vmin=0, vmax=7, extent=(0,N,N,0),interpolation='nearest')