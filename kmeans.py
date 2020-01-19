# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 04:25:29 2020

@author: Nakata Koya
"""

import SOLCSTestData as test
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans

N = int(math.sqrt(test.nodes.shape[0]))

data = list(test.dict_map_outliner_excluded.values())
indexes = list(test.dict_map_outliner_excluded.keys())
xy = np.unravel_index(indexes, (N, N))
features = list(zip(xy[0], xy[1]))

kmeans_model = KMeans(n_clusters=8).fit(features)

labels = kmeans_model.labels_

for label, cl, feature in zip(labels, data, features):
    print("class:", label, cl, "coord:", feature)

fig = plt.figure()

ax = fig.add_subplot(1,1,1)

ax.scatter(xy[0], xy[1], c=labels, marker="s")

ax.set_title('first scatter plot')
ax.set_xlabel('x')
ax.set_ylabel('y')