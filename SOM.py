# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 15:36:50 2019

@author: Nakata Koya
"""

import numpy as np
import os
import math
import pickle
#import pprint
import SOLCSConfig as conf
import SOLCSFigureGenerator as fg
import SOLCSAnalyseNodes as a
#import SOLCS_entropy as entropy
from tqdm import tqdm
#from itertools import groupby
from operator import itemgetter
#from numba.decorators import jit

ADBIT00 = [0,1,2]
ADBIT01 = [0,1,3]
ADBIT10 = [0,1,4]
ADBIT11 = [0,1,5]
ADBIT_IDX = {"BLACK":[0,1,2], "RED":[0,1,3], "GREEN":[0,1,4], "BLUE":[0,1,5]}

ADBIT_VALS = [[0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1]]
ACTIONS = [0,1]

class SOM():
    def __init__(self, teachers, N, upd_bit = conf.ADBIT00, head=None, seed=None, doesErrorCorrect = False):
        #seed setting
        if seed==None:
            None
        else: #seedが設定されている
            np.random.seed(seed)
        
        self.teachers = np.array(teachers)
        self.n_teacher = self.teachers.shape[0]
        self.head = head
        self.N = N
        self.upd_bit = upd_bit
        self.doesErrorCorrect = doesErrorCorrect
        
        #格子点の生成
        x, y = np.meshgrid(range(self.N), range(self.N))
        #二次元座標の一次元配列
        self.c = np.hstack((y.flatten()[:, np.newaxis],x.flatten()[:, np.newaxis]))
        #初期マップの生成 caution:11月11日まで行動もランダムに与えていた！
        self.nodes = np.round(np.random.rand(self.N*self.N, self.teachers.shape[1]))
        
        #正解行動の付与 note:このブロックを消せば行動はランダムで与えられる
        for cl in self.nodes:
            cl[-1] = getAns(cl)
            
        self.ininodes = np.copy(self.nodes)
        
        self.unique_dic_each_itr = {}
            
        #得点の付与
        #for cl in self.nodes:
        #    cl.append(1000*cl[-1])
            
        #entropy
        #self.entropy_list = []
            
    def train(self):
        print("training has started.")

        #hack
        for i, teacher in enumerate(tqdm(self.teachers, position=0, desc="te:"+str(conf.seed_teacher)+" tr"+str(conf.seed_train))):
            mu = self._best_matching_unit(teacher, all_index=True)
            bmu = np.unravel_index(mu[0], (self.N, self.N))
            d = np.linalg.norm(self.c - bmu, axis=1) #cとbmuのmap上での距離
            L = self._learning_ratio(i)
            S = self._learning_radius(i, d)
            
            #ユニークルールセット
            if i%1000 == 0:
                unique_dic_dic = a.unique_dic_dic(np.round(self.nodes))
                self.unique_dic_each_itr[i] = unique_dic_dic
            
            #BMUが複数現れたら代表以外のものを突然変異
            self.mutate(mu[1:])
            
            #teacher[self.head:-2] = 0 #先頭3ビット＋行動だけ更新する場合
            #todo:                                                                                            
            self.nodes[:,self.upd_bit] +=  L * S[:, np.newaxis] * (teacher[self.upd_bit] - self.nodes[:,self.upd_bit])

            #全ビット更新
            #self.nodes +=  L * S[:, np.newaxis] * (teacher - self.nodes)
            
            #誤り訂正　正解行動を付与
            #todo bottle neck
            if self.doesErrorCorrect:
                self.nodes[:,-1] = getAnsNodes(np.round(self.nodes))

        print("training has finished")
        return self.nodes
        
    def _best_matching_unit(self, teacher, all_index=False):
        if self.head == None:
            norms = np.linalg.norm(self.nodes - teacher, axis=1)
        else: #self.head != None
            #todo
            #idx = list(range(self.head)) + [-1]
            idx = self.upd_bit + [-1]
            nodes = self.nodes[:,idx]
            teacher = teacher[idx]
            #ここで1ビットだけ1000倍のオーダーだったらノルムにどう響く?
            norms = np.linalg.norm(nodes - teacher, axis=1)
        
        if all_index:
            #全てのインデックスを返す
            return np.where(norms == norms.min())[0]
        else:
            #最初に見つかったインデックスを一つだけ返す
            return np.argmin(norms) #normsを1次元にreshapeしたときのインデックス
            
        #return np.unravel_index(bmu,(self.N, self.N)) #N*N行列のargmin

    def _neighbourhood(self, t):#neighbourhood radious
        halflife = float(self.n_teacher/4) #for testing
        initial  = float(self.N/2)
        return initial*np.exp(-t/halflife)

    def _learning_ratio(self, t):
        halflife = float(self.n_teacher/4) #for testing
        initial  = 0.1
        return initial*np.exp(-t/halflife)

    def _learning_radius(self, t, d):
        # d is distance from BMU
        s = self._neighbourhood(t)
        return np.exp(-d**2/(2*s**2))
    
    def mutate(self, mu):
        def _mutate(node):
            #node[2:-1] = np.random.permutation(node[2:-1])
            node[2:-1] = np.random.rand(4)
            return node
        
        for idx in mu:
            self.nodes[idx] = _mutate(self.nodes[idx])
    
def generateMUXNodes(num_teachers, seed=None, k=2, P_sharp = 0, includeAns = False, includeRewards = False):
    #seed setting
    if seed==None:
        None
    else: #seedが設定されている
        np.random.seed(seed)
        
    teachers = []
    bits = k + 2**k
    for i in range(num_teachers):
        teacher = []
        
        #問題を生成
        for j in range(bits):
            if np.random.rand() < P_sharp:
                teacher.append("#")
            else:
                teacher.append(np.random.randint(2))
                
        #必要なら答えを付与
        if includeAns == True:
            teacher.append(getAns(teacher))
            
        #必要なら報酬を付与
        if includeRewards == True:
            teacher.append(getAns(teacher)*1000)

        teachers.append(teacher)
    return teachers

def getAns(bitArray):
    addbit = bitArray[:conf.k]
    #refbit = bitArray[k:]
    #cal = ""
    #正解行動
    #for x in range(len(addbit)):
    #    cal += str(int(addbit[x]))

    #cal = int(cal,2)
    #ans = refbit[cal]
    #return ans
        
    #return refbit[int(cal,2)]
    return bitArray[conf.k+int("".join(map(str, map(int, addbit))),2)]
    #return bitArray[conf.k+int("".join([str(int(x)) for x in addbit]),2)]

#act含む/含まない両方対応
def getAnsNodes(nodes): #nodes.shape must be [N*N, bits]
    #ansNodes = []
    #for cl in nodes:
    #    ansNodes.append(getAns(cl))
            
    #ansNodes = np.array(ansNodes)
    #return ansNodes.reshape(conf.N, conf.N)
    return np.array(list(map(getAns, nodes))) #mapで高速化できるか?
    #return np.array([getAns(cl) for cl in nodes])

class Main():
    def __init__(self, upd_bit=conf.ADBIT00):
        os.makedirs(conf.dirStr_result() ,exist_ok=True)            
        self.teachers = generateMUXNodes(seed=conf.seed_teacher, k=conf.k, includeAns=conf.includeAns, num_teachers=conf.num_teachers, includeRewards = conf.includeRewards)        
        self.som = SOM(self.teachers, N=conf.N, upd_bit=upd_bit, head=conf.head, seed=conf.seed_train, doesErrorCorrect = conf.doesErrorCorrect)

    def main(self):
        self.som.train() #som.nodes.shape = (N*N=100*100, bits=7)
        
        #結果をpickleに保存
        with open(conf.dirStr_result() + "\\" + "nodes.bin", "wb") as nodes:
            pickle.dump(self.som.nodes, nodes)        
                  
        with open(conf.dirStr_result() + "\\" + "ininodes.bin", "wb") as ininodes:
            pickle.dump(self.som.ininodes, ininodes)
           
        with open(conf.dirStr_result() + "\\" + "unique_dic_each_itr.bin", "wb") as dic:
            pickle.dump(self.som.unique_dic_each_itr, dic)
            
        a.save_unique_cls_dic_dic_as_txt(self.som.unique_dic_each_itr, conf.dirStr_result())
                    
        

        #how to load
        #with open("nodes.bin", "rb") as nodes:
        #   nodes = pickle.load(nodes)        

if __name__ == "__main__":
    main = Main()
    #main.main()
    #fg.FigureGenerater(dirStr_result = SOM.conf.dirStr_result()).genFig(doesShow=False)