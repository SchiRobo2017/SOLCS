# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 19:07:56 2020

@author: Nakata Koya
"""
import numpy as np
import pickle
import importlib
import SOLCSFigureGenerator as fg

""""norm: ord=1 (manhattan)"""

"""
path = ["C:/Users/Nakata Koya/python/XCSandSOM_git/exp_data/teacher1_train10_[0, 1, 5]_N24_in_norm_ord1_20200130104448",
        "C:/Users/Nakata Koya/python/XCSandSOM_git/exp_data/teacher1_train10_[0, 1, 4]_N24_in_norm_ord1_20200130104448",
        "C:/Users/Nakata Koya/python/XCSandSOM_git/exp_data/teacher1_train10_[0, 1, 3]_N24_in_norm_ord1_20200130104448",
        "C:/Users/Nakata Koya/python/XCSandSOM_git/exp_data/teacher1_train10_[0, 1, 2]_N24_in_norm_ord1_20200130104448"]
"""

#path = path[3]

#path = "C:/Users/Nakata Koya/python/XCSandSOM_git/exp_data/teacher1_train10_N24_all_bits_upd_20200130123328"

#path = "C:/Users/Nakata Koya/python/XCSandSOM_git/exp_data/teacher1_train10_N24_allbits_update_1001_20200131174016"

path = ["C:/Users/Nakata Koya/python/XCSandSOM_git/exp_data/teacher1_train10_[0, 1, 5]_N24/teacher1_train10_[0, 1, 5]_N24_20200130183943",
        "C:/Users/Nakata Koya/python/XCSandSOM_git/exp_data/teacher1_train10_[0, 1, 5]_N24/teacher1_train10_[0, 1, 4]_N24_20200130183943",
        "C:/Users/Nakata Koya/python/XCSandSOM_git/exp_data/teacher1_train10_[0, 1, 5]_N24/teacher1_train10_[0, 1, 3]_N24_20200130183943",
        "C:/Users/Nakata Koya/python/XCSandSOM_git/exp_data/teacher1_train10_[0, 1, 5]_N24/teacher1_train10_[0, 1, 2]_N24_20200130183943"]

path = ["C:/Users/Nakata Koya/python/XCSandSOM_git/exp_data/teacher1_train10_N24/teacher1_train10_[0, 1, 2]_N24_20200130183943", "C:/Users/Nakata Koya/python/XCSandSOM_git/exp_data/teacher1_train10_N24/teacher1_train10_[0, 1, 3]_N24_20200130183943", "C:/Users/Nakata Koya/python/XCSandSOM_git/exp_data/teacher1_train10_N24/teacher1_train10_[0, 1, 4]_N24_20200130183943", "C:/Users/Nakata Koya/python/XCSandSOM_git/exp_data/teacher1_train10_N24/teacher1_train10_[0, 1, 5]_N24_20200130183943"]

path = path[0]

fg_ = fg.FigureGenerater(path)

with open(path+"/nodes.bin", "rb") as nodes:
    nodes = pickle.load(nodes)
    
with open(path+"/unique_dic_each_itr.bin", "rb") as unique_dic_each_itr:
    unique_dic_each_itr = pickle.load(unique_dic_each_itr)
    
nodes_rounded = np.round(nodes)

inp_sequencial = fg_.inp_sequencial

map_new_input = fg.map_(nodes, inp_sequencial)

dict_map_outliner_excluded = {}
for i, cl in enumerate(map_new_input):
    if cl[0] != -1:
        dict_map_outliner_excluded[i] = cl