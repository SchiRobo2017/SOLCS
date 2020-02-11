# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 16:29:59 2020

@author: Nakata Koya
"""


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

#entropy
def H(lst):
    return -sum([x*math.log2(x) if x != 0 else 0 for x in lst])

pd.set_option('display.max_rows', None)

N = int(math.sqrt(test.nodes.shape[0]))
k = 8
fname = "clusters_by_kmeans_k" + str(k)

classifiers = list(test.dict_map_outliner_excluded.values())
indexes = list(test.dict_map_outliner_excluded.keys())
xy = np.unravel_index(indexes, (N, N))
coordinates = list(zip(xy[0], xy[1]))

kmeans_model = KMeans(n_clusters=k).fit(coordinates)

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

plt.savefig(test.path + "/" + fname + ".png")

entropy_list = []
for i in range(k):
    mask = (clusters == i)
    cluster_tmp = np.array(classifiers)[mask]
    actions = cluster_tmp[:,-1]
    frac = actions.shape[0]
    p_act1 = sum(actions)/frac
    p_act0 = 1.0 - p_act1
    entropy = abs(H([p_act0, p_act1]))
    entropy_list.append(entropy)
    
df_entropy = pd.DataFrame(entropy_list, columns=["entropy"])

tmp = list(zip(clusters, classifiers, coordinates))
tmp.sort(key=lambda x : x[0])

clustered_cls = pd.DataFrame(tmp, columns=["cluster", "classifier", "coordinate"])
clustered_cls = pd.merge(clustered_cls, df_entropy, left_on="cluster", right_index=True)
clustered_cls.index+=1
clustered_cls.to_csv(test.path + "/" + fname + ".csv", sep="\t")