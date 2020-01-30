# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 19:07:56 2020

@author: Nakata Koya
"""
import numpy as np
import pickle
import importlib
import SOLCSFigureGenerator as fg

path = "C:/Users/Nakata Koya/python/XCSandSOM_git/exp_data/teacher1_train10_[0, 1, 2]_N=5_refact_test_20200110182050"

path = "C:/Users/Nakata Koya/python/XCSandSOM_git/exp_data/teacher1_train10_[0, 1, 2]_N24_map_test_20200116182201"

path = "C:/Users/Nakata Koya/python/XCSandSOM_git/exp_data/teacher1_train10_[0, 1, 2]_N24_with_final_input_20200119235213"

path = "C:/Users/Nakata Koya/python/XCSandSOM_git/exp_data/teacher1_train10_[0, 1, 2]_N24_L1_norm_test_20200120034552"

path = "C:/Users/Nakata Koya/python/XCSandSOM_git/exp_data/teacher1_train10_[0, 1, 2]_N24_L1_norm_test_20200120034552"

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