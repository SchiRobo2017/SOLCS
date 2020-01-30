# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 04:25:29 2020

@author: Nakata Koya
"""

import math
import SOLCSTestData as test
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

pd.set_option('display.max_rows', None)

N = int(math.sqrt(test.nodes.shape[0]))

classifiers = list(test.dict_map_outliner_excluded.values())
indexes = list(test.dict_map_outliner_excluded.keys())
xy = np.unravel_index(indexes, (N, N))
coordinates = list(zip(xy[0], xy[1]))

kmeans_model = KMeans(n_clusters=8).fit(coordinates)

clusters = kmeans_model.labels_

#for cluster, cl, coord in zip(clusters, classifiers, coordinates):
#    print("class:", cluster, cl, "coord:", coord)

fig = plt.figure(figsize=(8,8))

ax = fig.add_subplot(1,1,1)

ax.scatter(xy[1], -xy[0], c=clusters, marker="s")

plt.grid(True)
#plt.xlim(-1, N)
#plt.ylim(N, -1)

ax.set_title('clusters in SOM')
ax.set_xlabel('x')
ax.set_ylabel('y')

plt.savefig(test.path + "/clusters_in_som.png")

tmp = list(zip(clusters, classifiers, coordinates))
tmp.sort(key=lambda x : x[0])
clustered_cls = pd.DataFrame(tmp, columns=["cluster", "classifier", "coordinate"])
clustered_cls.index+=1
clustered_cls.to_csv(test.path + "/clustered_cls.csv", sep="\t")
