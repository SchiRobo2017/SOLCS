# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 15:36:50 2019

@author: Nakata Koya
"""

import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import math
import pickle

class SOM():
    def __init__(self, teachers, head, N, seed=None):
        #seed setting
        if seed==None:
            None
        else: #seedが設定されている
            np.random.seed(seed)
        
        self.teachers = np.array(teachers)
        self.n_teacher = self.teachers.shape[0]
        self.head = head
        self.N = N
        
        x, y = np.meshgrid(range(self.N), range(self.N)) #格子点の生成
        self.c = np.hstack((y.flatten()[:, np.newaxis],
                            x.flatten()[:, np.newaxis])) #座標の配列に変換
        self.nodes = np.round(np.random.rand(self.N*self.N,
                                    self.teachers.shape[1])) #初期マップの生成
        self.ims = []

    def train(self):
        print("training has started.")

        for i, teacher in enumerate(self.teachers):
            bmu = self._best_matching_unit(teacher)
            d = np.linalg.norm(self.c - bmu, axis=1) #cとbmuのmap上での距離
            L = self._learning_ratio(i)
            S = self._learning_radius(i, d)
            #teacher[self.head:-2] = 0 #先頭3ビット＋行動だけ更新する場合
            self.nodes +=  L * S[:, np.newaxis] * (teacher - self.nodes)
            
            if i%(len(self.teachers)/100*10)==0:
                print("progress : " + str(i/len(self.teachers)*100) + "%")

        print("training has finished")
        return self.nodes
    
    def _distance(self, x, axis = 1): #usage: distance(a-b)
        # #含む距離
        def calc_dist(x):
            dist = 0
            for elm in x:
                if elm == "#":
                    dist += 0
                elif elm == 0:
                    dist += 0
                elif abs(elm) == 1:
                    dist += 1
            return dist 
        print(calc_dist(x))
        return calc_dist(x)
        
    def _best_matching_unit(self, teacher):
        if self.head == None:
            norms = np.linalg.norm(self.nodes[:, :self.head] - teacher[:self.head], axis=1)
        else: #self.head != None
            idx = list(range(self.head)) + [-1]
            nodes = self.nodes[:,idx]
            teacher = teacher[idx]
            norms = np.linalg.norm(nodes - teacher, axis=1)

        bmu = np.argmin(norms) #normsを1次元にreshapeしたときのインデックス
        return np.unravel_index(bmu,(self.N, self.N)) #N*N行列のargmin

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

def generateMUXNodes(num_teachers, seed=None, k=2, P_sharp = 0, includeAns = False):
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
            teacher.append(getAns(teacher, k))

        teachers.append(teacher)
    return teachers

def getAns(bitArray, k=2):
    addbit = bitArray[:k]
    refbit = bitArray[k:]
    cal = ""
    #正解行動
    for x in range(len(addbit)):
        cal += str(int(addbit[x]))

    cal = int(cal,2)
    ans = refbit[cal]
    return ans

#act含む/含まない両方対応
def getAnsNodes(nodes, k=2): #nodes.shape must be [N*N, bits]
    ansNodes = []
    N = int(math.sqrt(nodes.shape[0])) #edge length of the map
    for cl in nodes:
        ansNodes.append(getAns(cl))
            
    ansNodes = np.array(ansNodes)
    return ansNodes.reshape(N, N)

def getColoredNodes(nodes, k=2, color="gray"): #nodes.shape must be [N*N, bits]
    Max = k + k**2
    N = int(math.sqrt(nodes.shape[0])) #edge length of the map
    coloredNodes = []
    if color=="colored":
        for cl in nodes:
            addBitsArray = cl[:k]
            refBitsArray = cl[k:-1]
            addBits = [str(int(i)) for i in addBitsArray]
            addBits = "".join(addBits)
            ansBit = refBitsArray[int(addBits,2)]
            if addBits=="00": #黒
                if ansBit == 1:
                    coloredNodes.append([0,0,0])
                else:
                    coloredNodes.append([128,128,128])    
            elif addBits=="01": #R
                if ansBit == 1:
                    coloredNodes.append([128,0,0])
                else:
                    coloredNodes.append([255,0,0])
            elif addBits=="10": #G
                if ansBit == 1:
                    coloredNodes.append([0,128,0])
                else:
                    coloredNodes.append([0,255,0])
            elif addBits=="11": #B
                if ansBit == 1:
                    coloredNodes.append([0,0,128])
                else:
                    coloredNodes.append([0,0,255])

        coloredNodes = np.array(coloredNodes, dtype = np.uint8)
        return coloredNodes.reshape(N, N, 3)
    
    elif color == "bits-scale":
        for cl in nodes:        
            coloredNodes.append(np.sum(cl)/Max)
            
        coloredNodes = np.array(coloredNodes, dtype = np.uint8)
        return coloredNodes.reshape(N, N)
    elif color == "bits2decimal-scale":
        for cl in nodes:
            cljoined = [str(int(i)) for i in cl]
            cljoined = cljoined[:k+k**2] #act bitを除外
            cljoined = "".join(cljoined)
            clIntScale = int(cljoined,2) #2進数の2
            coloredNodes.append(clIntScale)
            #coloredNodes.append(int("".join([str(int(i)) for i in cl]))) #一行で書けばこう
            
        #coloredNodes = np.array(coloredNodes, dtype = np.uint8)
        coloredNodes = np.array(coloredNodes)
        return coloredNodes.reshape(N, N)
    else:
        raise ValueError("colorに渡す引数が間違ってるよ")

    raise ValueError("colorに渡す引数が間違ってるよ")


def main():
    seed_teacher = 10
    seed_train = 10
    N = 100
    k = 2
    includeAns = True
    bits = k + 2**k
    if includeAns==True:
        bits+=1
    num_teachers = 10000 #default=10000 収束する   
    
    dt_now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    dirStr_result = "exp_data\\seed" + str(seed_train) #todo:命名規則の統一(適当に名前つけたので)
    os.makedirs(dirStr_result ,exist_ok=True)
        
    teachers = generateMUXNodes(seed=seed_teacher, k=2, includeAns=includeAns,
                                num_teachers=num_teachers)
    
    som = SOM(teachers, head=3, N=N, seed=None)
    
    iniNodes = getColoredNodes(som.nodes,
                               color="bits2decimal-scale")
    iniNodesColored = getColoredNodes(np.round(som.nodes),
                                      color="colored")
    iniCorrectActNodes = getAnsNodes(np.round(som.nodes)).reshape(N,N) #initial nodes of ansers
    
    #plt.figure()
    #plt.imshow(iniCorrectActNodes, cmap="gray", vmin=0, vmax=1, interpolation="none")
    #plt.title("initial map of actions")
    #plt.savefig(dirStr_result + "\\initial map of actions.png")
    
    #plt.figure()
    #plt.imshow(iniNodes, cmap="gray", vmin=0, vmax=63, interpolation="none")
    #plt.title("initial map of condition by 64 scale")
    #plt.savefig(dirStr_result + str(seed) + "\\initial map of condition by 64 scale")
    
    #plt.figure()
    #plt.imshow(iniNodesColored, cmap="gray", vmin=0, vmax=255, interpolation="none")
    #plt.title("initial map colored by address bit")
    #plt.savefig(dirStr_result + "\\initial map colored by address bit")
    
    som.train() #som.nodes.shape = (N*N=100*100, bits=7)
    
    #結果をpickleに保存
    with open(dirStr_result + "\\" + "nodes.bin", "wb") as nodes:
        pickle.dump(som.nodes, nodes)

    #how to load
    #with open("nodes.bin", "rb") as nodes:
    #   nodes = pickle.load(nodes)
    
    actNodesRealNum = som.nodes[:,-1].reshape(N, N)
    actNodes = np.round(actNodesRealNum)
    correctActNodes = getAnsNodes(np.round(som.nodes))
    afterNodesRounded = getColoredNodes(np.round(som.nodes),
                                        color="bits2decimal-scale")
    
    afterNodesReverse = np.round(som.nodes)[:,0:-1]
    afterNodesReverse = getColoredNodes(afterNodesReverse[:,::-1], color="bits2decimal-scale")
    
    afterNodesSeparated = afterNodesRounded.copy()
    afterNodesColored = getColoredNodes(np.round(som.nodes), color="colored")
    
    """
    plt.figure()
    plt.imshow(actNodesR, cmap="gray", vmin=0, vmax=1,
               interpolation="none")
    plt.title("map of action part after leaning(continuous value)")
    plt.savefig(dirStr_result + 
                "\\map of action part after leaning(countinuous value)")
    """
    
    plt.figure()
    plt.imshow(actNodes, cmap="gray", vmin=0, vmax=1,
               interpolation="none")
    plt.title("map of action part after leaning")
    plt.savefig(dirStr_result +
                "\\map of action part after leaning" 
                + dt_now)
    
    plt.figure()
    plt.imshow(correctActNodes, cmap="gray", vmin=0, vmax=1, interpolation="none")
    plt.title("map of correct action part after leaning")
    plt.savefig(dirStr_result +
                "\\map of correct action part after leaning"
                + dt_now)
    
    plt.figure()
    plt.imshow(afterNodesRounded, cmap="gray", vmin=0, vmax=63, interpolation="none")
    plt.title("map of condition part after learning")
    plt.colorbar()
    plt.savefig(dirStr_result +
                "\\map of condition part after learning"
                + dt_now)
    
    plt.figure()
    plt.imshow(afterNodesReverse , cmap="gray", vmin=0, vmax=63, interpolation="none")
    plt.title("map of condition part after learning(reversed value)")
    plt.colorbar()
    plt.savefig(dirStr_result +
                "\\map of condition part after learning(reversed value)" 
                + dt_now)
    
    #afterNodesSeparatedの値を行動の0,1に応じて色分け
    for i, row in enumerate(correctActNodes):
        for j, ans in enumerate(row):
            if ans == 1.0:
                None
            elif ans == 0.0:
                val = afterNodesSeparated[i,j]
                afterNodesSeparated[i,j] = -val
                
    plt.figure()
    plt.imshow(afterNodesSeparated, cmap="PuOr", vmin=-64, vmax=63, interpolation="none")
    plt.title("map of condition part separated by action")
    plt.colorbar()
    plt.savefig(dirStr_result +
                "\\map of condition part separated by action" 
                + dt_now)
    
    plt.figure()
    plt.imshow(afterNodesColored, cmap="gray", vmin=0, vmax=255, interpolation="none")
    plt.title("map after learning coloerd by address bit")
    plt.savefig(dirStr_result +
                "\\map after learning coloerd by address and act" 
                + dt_now)    
    
    plt.show()
    
    black00_0 = np.round(som.nodes).reshape(100,100,7)[40:50,85:95,:]
    black00_1 = np.round(som.nodes).reshape(100,100,7)[0:10,0:10,:]
    
    red01_0 = np.round(som.nodes).reshape(100,100,7)[0:10,50:60,:]
    red01_1 = np.round(som.nodes).reshape(100,100,7)[0:10,30:40,:]
    
    green10_1 = np.round(som.nodes).reshape(100,100,7)[40:50,30:40,:]
    green10_0 = np.round(som.nodes).reshape(100,100,7)[40:50,30:40,:]
    
    blue11_0 = np.round(som.nodes).reshape(100,100,7)[70:80,89:99,:]
    blue11_1 = np.round(som.nodes).reshape(100,100,7)[89:99,30:40,:]
    
    black00_0 = np.unique(black00_0.reshape(100,7),axis=0)
    black00_1 = np.unique(black00_1.reshape(100,7),axis=0)
    
    red01_0 = np.unique(red01_0.reshape(100,7),axis=0)
    red01_1 = np.unique(red01_1.reshape(100,7),axis=0)
    
    green10_0 = np.unique(green10_0.reshape(100,7),axis=0)
    green10_1 = np.unique(green10_1.reshape(100,7),axis=0)
    
    blue11_0 = np.unique(blue11_0.reshape(100,7),axis=0)
    blue11_1 = np.unique(blue11_1.reshape(100,7),axis=0)
    
    red01_bound = np.round(som.nodes).reshape(100,100,7)[0:30,40:50,:]
    green10_bound = np.round(som.nodes).reshape(100,100,7)[50:55,0:15,:]
    blue11_bound = np.round(som.nodes).reshape(100,100,7)[89:99,10:15,:]
    
    red01_bound = np.unique(red01_bound.reshape(300,7), axis=0)
    green10_bound = np.unique(green10_bound.reshape(75,7), axis=0)
    blue11_bound = np.unique(blue11_bound.reshape(50,7), axis=0)
    
    print("end of program")

if __name__ == "__main__":
    main()