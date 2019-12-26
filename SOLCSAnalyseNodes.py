# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 08:38:24 2019

@author: Nakata Koya
"""
import matplotlib.pyplot as plt
import SOLCSFigureGenerator as fg
import pickle
import numpy as np
from itertools import groupby
from operator import itemgetter

cl_sharp_expression = [0,0,0,"#","#","#",0]

class AnalyseNodes():
    #load nodes data
    def __init__(self, resultDirStr, is_relative_path=True):
        if resultDirStr == "":
            print("test execution")
            
        if is_relative_path==True:
            self.resultDirStr = "exp_data\\" + resultDirStr
            with open(self.resultDirStr +  "\\nodes.bin", "rb") as nodes:
                self.nodes = pickle.load(nodes)
        else:
            self.resultDirStr = resultDirStr
            with open(self.resultDirStr, "rb") as nodes:
                self.nodes = pickle.load(nodes)
            
    def extractWithColor(self, color): #クラスタの抽出
        cluster = []
        for cl in np.round(self.nodes):
            if (color == cl[:2]).all():
                cluster.append(cl)
        return np.array(cluster)
    
    def uniqueClsDicByCounts(self, unique_counts, unique_cls):
        unique_dic = []
        
        #登場回数と対応する分類子のタプルのリスト
        for count, cl in zip(unique_counts, unique_cls):
            unique_dic.append((count,cl))
            
        #登場回数でソート
        return sorted(unique_dic, key=itemgetter(0))
    
    def printClsGroupby(self, unique_dic):
        for (count, cl_group) in groupby(unique_dic, key=itemgetter(0)):
            print("count:" + str(count))
            for cl in cl_group:
                print(cl[1])
        print("\n")
        
    def matchingUnits(self, compare, nodes, allIdx=True):
        norms = np.linalg.norm(np.round(self.nodes) - compare, axis=1)
        
        if allIdx == False:
            #最初に見つかったインデックスを一つだけ返す
            bmu = np.argmin(norms) #normsを1次元にreshapeしたときのインデックス
        else:
            #全てのインデックスを返す
            bmu = np.where(norms == norms.min())
            
        return np.unravel_index(bmu,(100, 100)) #hack: todo: magic number #N*N行列のargmin
        
    def mapping(self, mappingDataList):
        mapped = np.full((100, 100, 7), -1) #hack: todo: magic number
        for i, cl in enumerate(mappingDataList):
            idx = self.matchingUnits(nodes=self.nodes, compare=cl, allIdx=True)
            mapped[idx] = cl
        
        return mapped.reshape(100*100, 7) #hack: todo: magic number
    
    def general_cls_from_dontcare_expression(self, cl_includes_sharp = [1,1,"#","#","#",1,1]):
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
    
    def extract_cls_by_action(self, unique_cls, action=0):
        for cl in unique_cls:
            if cl[-1] == action:
                print(cl)
            
if __name__ == "__main__":
    dirStr = input("input dir pass like \"seed + xx\":")
    a = AnalyseNodes(dirStr, is_relative_path=False)
    
    """
    classifiers each of adress part or bound data on map
    """
    black = [0,0]
    red = [0,1]
    green = [1,0]
    blue = [1,1]
    
    black00 = a.extractWithColor(black)
    black00_unique, black00_unique_counts = np.unique(black00, return_counts=True,  axis=0)    
    black00_unique_dic = a.uniqueClsDicByCounts(black00_unique_counts, black00_unique)
            
    red01 = a.extractWithColor(red)
    red01_unique, red01_unique_counts = np.unique(red01, return_counts=True,  axis=0)
    red01_unique_dic = a.uniqueClsDicByCounts(red01_unique_counts, red01_unique)
    
    green10 = a.extractWithColor(green)
    green10_unique, green10_unique_counts = np.unique(green10, return_counts=True,  axis=0)
    green10_unique_dic = a.uniqueClsDicByCounts(green10_unique_counts, green10_unique)
    
    blue11 = a.extractWithColor(blue)
    blue11_unique, blue11_unique_counts = np.unique(blue11, return_counts=True,  axis=0)
    blue11_unique_dic = a.uniqueClsDicByCounts(blue11_unique_counts, blue11_unique)
    
    #print("seed_teacher = 10, seed_train = None") #hack: magic number
    print("black00: unique cls =", len(black00_unique_dic), ", total cls =", sum(black00_unique_counts))
    a.printClsGroupby(black00_unique_dic)
    print("red01: unique cls =", len(red01_unique_dic), ", total cls =", sum(red01_unique_counts))
    a.printClsGroupby(red01_unique_dic)
    print("green10: unique cls =", len(green10_unique_dic), ", total cls =", sum(green10_unique_counts))
    a.printClsGroupby(green10_unique_dic)
    print("blue11: unique cls =", len(blue11_unique_dic), ", total cls =", sum(blue11_unique_counts))
    print("blue11:")
    a.printClsGroupby(blue11_unique_dic)
    
    
    #mapping_data = [[1,1,0,0,0,1,1]]
    mapping_data = a.general_cls_from_dontcare_expression(cl_includes_sharp = cl_sharp_expression)
    nodes_mapped_new_input = a.mapping(mapping_data)
    
    nodes_mapped_new_input_colored = fg.SOM.getColoredNodes(nodes_mapped_new_input, color="colored")

    plt.figure()
    plt.imshow(nodes_mapped_new_input_colored, cmap="gray", vmin=0, vmax=255, interpolation="none")
    plt.title("map of new classifier input")
    plt.show()
    
    """
    plt.savefig(dirStr_result +
                "\\map of new classifier input" 
                + dt_now)
    """