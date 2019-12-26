# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 17:59:06 2019

@author: Nakata Koya
"""

import math
import pprint
import numpy as np
import pickle

from tqdm import tqdm

ADBIT_IDX = {"BLACK":[0,1,2], "RED":[0,1,3], "GREEN":[0,1,4], "BLUE":[0,1,5]}

def fraction(nodes_rounded, adbits, actions):
    ret_dic = {}
    #dic["key"] = "value"    
    
    for adbit in adbits:
        #print(adbit)
        nodes_adbit = _extractWithAdbit(adbit, nodes_rounded)
        for act in actions:
            #print([act])
            nodes_act = _extractWithAct(act, nodes_adbit)
            
            count_adbit = len(nodes_adbit)
            count_act = len(nodes_act)
            
            frac = count_act / count_adbit
            #print(frac)
            ret_dic[(tuple(adbit), act)] = frac

    return ret_dic

def entropy(nodes_rounded, adbits, actions):
    ret_dic = {}
    
    for adbit in adbits:
        nodes_adbit = _extractWithAdbit(adbit, nodes_rounded)
        entropy = 0
        for act in actions:
            nodes_act = _extractWithAct(act, nodes_adbit)
            
            count_adbit = len(nodes_adbit)
            count_act = len(nodes_act)
            
            p = count_act / count_adbit
            
            if p == 0.0:
                self_entropy = -p*math.log2(0.01)
            else:
                self_entropy = -p*math.log2(p)
            entropy += self_entropy
            
        ret_dic[tuple(adbit)] = entropy
        
    return ret_dic    

def _extractWithAdbit(adbit, nodes_rounded): #adbitが一致するノードの抽出
    cluster = []
    for cl in nodes_rounded:
        if (adbit == cl[adbit_idx]).all():
            cluster.append(cl)
    return np.array(cluster)

def _extractWithAct(act, nodes_rounded): #adbitが一致するノードの抽出
    cluster = []
    for cl in nodes_rounded:
        if act == cl[-1]:
            cluster.append(cl)
    return np.array(cluster)

if __name__ == "__main__":
    adbits_vals = [[0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1]]
    actions = [0,1]
    
    resultDirStr = input("input dir path:")
    if resultDirStr == "":
        resultDirStr = "exp_data\\debug_and_error_correct_with_updating_map\\teacher1_train10_error_corrected"
        
    print(resultDirStr)
    with open(resultDirStr, "rb") as nodes:
        nodes = pickle.load(nodes)
        
    #print(nodes.shape)
    
    nodes = np.round(nodes)
    #print(nodes)
    
    #nodes_extracted_with_adbit = _extractWithAdbit([0,0,0], nodes)
    #print(adbit00)
    
    fractions = fraction(nodes, adbits_vals, actions)
    entropy_ = entropy(nodes, adbits_vals, actions)
    pprint.pprint(fractions)
    pprint.pprint(entropy_)