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

arr_1 = np.random.normal(scale=1.0, size=(100, 2))
arr_2 = np.random.normal(scale=2.0, size=(300, 2))
arr_3 = np.random.normal(scale=1.5, size=(200, 2))
arr_4 = np.random.normal(scale=1.2, size=(50, 2))
arr_2[:, 0] += 10
arr_3[:, 0] += 4
arr_3[:, 1] += -7
arr_4[:, 0] += 11
arr_4[:, 1] += -9
X = np.concatenate([
    arr_1,
    arr_2,
    arr_3,
    arr_4,
])

data = list(test.dict_map_outliner_excluded.values())
labels = list(test.dict_map_outliner_excluded.keys())
#df = pd.DataFrame(data)
#Z = linkage(df,method="ward",metric="euclidean")

#r = dendrogram(Z, p=10, truncate_mode="level", labels=labels)

#t = 0.7*max(Z[:,2])
#c = fcluster(Z, t=8, criterion="maxclust")

#x, y = np.meshgrid(range(N), range(N))

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