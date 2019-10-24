# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 08:38:24 2019

@author: Nakata Koya
"""
import pickle
import numpy as np
from itertools import groupby
from operator import itemgetter

class AnalyseNodes():
    #load nodes data
    def __init__(self, resultDirStr):
        self.resultDirStr = "exp_data\\" + resultDirStr
        with open(self.resultDirStr +  "\\nodes.bin", "rb") as nodes:
            self.nodes = pickle.load(nodes)
            
if __name__ == "__main__":
    dirStr = input("input dir pass like \"seed + xx\":")
    a = AnalyseNodes(dirStr)
    
    """
    classifiers each of adress part or bound data on map
    """
    black00 = []
    for cl in np.round(a.nodes):
        if ([0,0] == cl[:2]).all():
            black00.append(cl)
    black00 = np.array(black00)
    black00_unique, black00_unique_counts = np.unique(black00, return_counts=True,  axis=0)
    #black00_unique[np.argsort(black00_unique_counts)]    
    #black00_unique_dic = dict(zip(black00_unique_counts, black00_unique))
    black00_unique_dic = []
    for count, cl in zip(black00_unique_counts, black00_unique): #まずタプルのリストを生成
        black00_unique_dic.append((count, cl))

    #タプルのリストを分類子の登場回数=countでソート
    black00_unique_dic = sorted(black00_unique_dic, key=itemgetter(0))
    
    #キー(count)に対する分類子のグループごとに一覧表示
    for (count, cl_group) in groupby(black00_unique_dic, key=itemgetter(0)):
        print("count:" + str(count))
        for cl in cl_group:
            print(cl)
    
    red01 = []
    for cl in np.round(a.nodes):
        if ([0,1] == cl[:2]).all():
            red01.append(cl)
    red01 = np.array(red01)
    red01_unique = np.unique(red01, axis=0)
    
    green10 = []
    for cl in np.round(a.nodes):
        if ([1,0] == cl[:2]).all():
            green10.append(cl)
    green10 = np.array(green10)
    green10_unique = np.unique(green10, axis=0)
    
    blue11 = []
    for cl in np.round(a.nodes):
        if ([1,1] == cl[:2]).all():
            blue11.append(cl)
    blue11 = np.array(blue11)
    blue11_unique = np.unique(blue11, axis=0)