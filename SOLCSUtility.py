# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 18:53:26 2020

@author: Nakata Koya
"""
import math
import numpy as np

def getAns(bitArray, k=2):
    addbit = bitArray[:k]
    #refbit = bitArray[k:]
    #cal = ""
    #正解行動
    #for x in range(len(addbit)):
    #    cal += str(int(addbit[x]))

    #cal = int(cal,2)
    #ans = refbit[cal]
    #return ans
        
    #return refbit[int(cal,2)]
    return bitArray[k+int("".join(map(str, map(int, addbit))),2)]
    #return bitArray[conf.k+int("".join([str(int(x)) for x in addbit]),2)]

def best_matching_unit(nodes, cl, all_idx=False):
    norms = np.linalg.norm(nodes - cl, axis=1)

    if all_idx == False:
        #最初に見つかったインデックスを一つだけ返す
        bmu = np.argmin(norms) #normsを1次元にreshapeしたときのインデックス
    else:
        #全てのインデックスを返す
        bmu = np.where(norms == norms.min())
        
    N = int(math.sqrt(nodes.shape[0]))
        
    return np.unravel_index(bmu,(N, N)) #N*N行列のargmin

def general_cls_from_dontcare_expression(cl_includes_sharp = [1,1,"#","#","#",1,1]):
    count_sharp = cl_includes_sharp.count("#")
    num_cls = pow(2,count_sharp)
    
    replace_bit_arr = []
    for i in range(num_cls):
        replace_bit_arr.append(list(map(int, list(bin(i)[2:].zfill(count_sharp)))))
        
    #print(replace_bit_arr)
    
    idx_sharps = [i for i, x in enumerate(cl_includes_sharp) if x == "#"]
    
    general_cls = []
    for replace_bit in replace_bit_arr:
        cl_replaced = np.array(cl_includes_sharp)
        cl_replaced[idx_sharps] = replace_bit
        general_cls.append(cl_replaced.tolist())
        
    return [list(map(int, cl)) for cl in general_cls]