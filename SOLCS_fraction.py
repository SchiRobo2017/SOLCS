# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 17:59:06 2019

@author: Nakata Koya
"""

import numpy as np
import pickle

from tqdm import tqdm

def fraction(nodes_rounded, adbits, actions):
    for adbit in tqdm(adbits):
        print(adbit)
        nodes_adbit = extractWithAdbit(adbit, nodes_rounded)
        for act in actions:
            print([act])
            nodes_act = extractWithAct(act, nodes_adbit)
            
            count_adbit = len(nodes_adbit)
            count_act = len(nodes_act)
            
            print(count_act / count_adbit)
        
    #nodes_adbit = extractWithAdbit(adbit, nodes_rounded)
    #nodes_act = extractWithAct(act, nodes_adbit)
    
    #count_adbit = len(nodes_adbit)
    #count_act = len(nodes_act)
    
    #return count_act / count_adbit

def extractWithAdbit(adbit, nodes_rounded): #adbitが一致するノードの抽出
    cluster = []
    for cl in nodes_rounded:
        if (adbit == cl[:len(adbit)]).all():
            cluster.append(cl)
    return np.array(cluster)

def extractWithAct(act, nodes_rounded): #adbitが一致するノードの抽出
    cluster = []
    for cl in nodes_rounded:
        if act == cl[-1]:
            cluster.append(cl)
    return np.array(cluster)

if __name__ == "__main__":
    adbits = [[0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1]]
    actions = [0,1]
    resultDirStr = "exp_data\\debug_and_error_correct_with_updating_map\\teacher1_train10_error_corrected"
    with open(resultDirStr +  "\\nodes.bin", "rb") as nodes_bin:
        nodes = pickle.load(nodes_bin)
        
    #print(nodes.shape)
    
    nodes = np.round(nodes)
    #print(nodes)
    
    #nodes_extracted_with_adbit = extractWithAdbit([0,0,0], nodes)
    #print(adbit00)
    
    fractions = fraction(nodes, adbits, actions)
    print(fractions)