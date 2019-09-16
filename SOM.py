# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 15:36:50 2019

@author: Nakata Koya
"""

import numpy as np
import matplotlib.pyplot as plt

class SOM():

    def __init__(self, teachers, head, N):
        self.teachers = np.array(teachers)
        self.n_teacher = self.teachers.shape[0]
        self.head = head
        self.N = N
        
        x, y = np.meshgrid(range(self.N), range(self.N)) #格子点の生成
        self.c = np.hstack((y.flatten()[:, np.newaxis],
                            x.flatten()[:, np.newaxis])) #座標の配列に変換
        self.nodes = np.round(np.random.rand(self.N*self.N,
                                    self.teachers.shape[1]))
        self.ims = []

    def train(self):
        print("training has started.")

        for i, teacher in enumerate(self.teachers):
            if not(self.head == None):
                idx = list(range(self.head)) + [-1]
                bmu = self._best_matching_unit(teacher[idx])
            else:
                bmu = self._best_matching_unit(teacher[:self.head])
            d = np.linalg.norm(self.c - bmu, axis=1) #cとbmuのmap上での距離
            L = self._learning_ratio(i)
            S = self._learning_radius(i, d)
            teacher[self.head:-2] = 0 #先頭3ビット＋行動だけ学習させる場合
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
        """
        if not(self.head == None):
            idx = list(range(self.head)) + [-1]
            nodes = self.nodes[:,idx]
            teacher = teacher[idx]
            norms = np.linalg.norm(nodes - teacher, axis=1)
        else:
            norms = np.linalg.norm(self.nodes[:, :self.head] - teacher[:self.head], axis=1)
        """
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

def generateMUXNodes(num_teachers, k=2, P_sharp = 0, includeAns = False):
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
    addbit = bitArray[0:k]
    refbit = bitArray[k:]
    cal = ""
    #正解行動
    for x in range(len(addbit)):
        #cal += str(int(addbit[x]))
        cal += str(int(addbit[x]))
    cal = int(cal,2)
    ans = refbit[cal]
    return ans

#act含む/含まない両方対応
def getAnsNodes(nodes, k=2):
    ansNodes = []
    for row in nodes:
        for elm in row:
            addbit = elm[0:k]
            refbit = elm[k:k+k**2]
            cal = ""
            #正解行動
            for x in range(len(addbit)):
                #cal += str(int(addbit[x]))
                cal += str(int(addbit[x]))
            cal = int(cal,2)
            ans = refbit[cal]
            ansNodes.append(ans)
            
    ansNodes = np.array(ansNodes)
    return ansNodes.reshape(nodes.shape[0], nodes.shape[1], 1)

def getColoredNodes(nodes, k=2, color="gray"):
    Max = k + k**2
    coloredNodes = []
    if color=="colored":
        for cl in nodes:
            #addBitsArray = cl[:k].astype(int)
            addBitsArray = cl[:k]
            addBits = [str(int(i)) for i in addBitsArray]
            addBits = "".join(addBits)
            if addBits=="00": #白
                coloredNodes.append([0,0,0])
            elif addBits=="01": #R
                coloredNodes.append([255,0,0])
            elif addBits=="10": #G
                coloredNodes.append([0,255,0])
            elif addBits=="11": #B
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

seed = 10
N = 100
k = 2
includeAns = True
bits = k + 2**k
if includeAns==True:
    bits+=1
num_teachers = 10000 #default=10000 収束する   
    
np.random.seed(seed)

teachers = generateMUXNodes(k=2, includeAns=includeAns,
                            num_teachers=num_teachers)
som = SOM(teachers, head=3, N=N)

m = som.nodes.reshape((N,N,bits)) #initial nodes of cl
m1 = np.round(m)
iniNodes = getColoredNodes(som.nodes,
                           color="bits2decimal-scale")
iniNodesColored = getColoredNodes(np.round(som.nodes),
                                  color="colored")
iniCorrectActNodes = getAnsNodes(m1).reshape(N,N) #initial nodes of ansers

#plt.figure()
#plt.imshow(iniCorrectActNodes, cmap="gray", vmin=0, vmax=1, interpolation="none")
#plt.title("initial map of actions")
#plt.savefig("exp_data\\seed" + str(seed) + "\\initial map of actions.png")

#plt.figure()
#plt.imshow(iniNodes, cmap="gray", vmin=0, vmax=63, interpolation="none")
#plt.title("initial map of condition by 64 scale")
#plt.savefig("exp_data\\seed" + str(seed) + "\\initial map of condition by 64 scale")

#plt.figure()
#plt.imshow(iniNodesColored, cmap="gray", vmin=0, vmax=255, interpolation="none")
#plt.title("initial map colored by address bit")
#plt.savefig("exp_data\\seed" + str(seed) + "\\initial map colored by address bit")

som.train()

#actNodes = np.round(som.nodes[:,-1].reshape(N, N))
actNodes = getAnsNodes(np.round(som.nodes.reshape(N,N,bits))).reshape(N,N)
correctActNodes = getAnsNodes(np.round(som.nodes.reshape(N,N,bits))).reshape(N,N)
afterNodesRounded = getColoredNodes(np.round(som.nodes),
                                    color="bits2decimal-scale") #丸めると不思議な模様が！
afterNodesSeparated = afterNodesRounded.copy()
afterNodesColored = getColoredNodes(np.round(som.nodes), color="colored")

plt.figure()
plt.imshow(actNodes, cmap="gray", vmin=0, vmax=1,
           interpolation="none")
plt.title("map of action part after leaning")
plt.savefig("exp_data\\seed" + str(seed) + 
            "\\map of action part after leaning")

plt.figure()
plt.imshow(correctActNodes, cmap="gray", vmin=0, vmax=1, interpolation="none")
plt.title("map of correct action part after leaning")
plt.savefig("exp_data\\seed" + str(seed) +
            "\\map of correct action part after leaning")

plt.figure()
plt.imshow(afterNodesRounded, cmap="Blues", vmin=0, vmax=63, interpolation="none")
plt.title("map of condition part after learning")
plt.colorbar()
plt.savefig("exp_data\\seed" + str(seed) + "\\map of condition part after learning")

for i, row in enumerate(actNodes):
    for j, ans in enumerate(row):
        if ans == 1.0:
            None
        elif ans == 0.0:
            val = afterNodesSeparated[i,j]
            afterNodesSeparated[i,j] = -val
            
plt.figure()
plt.imshow(afterNodesSeparated, cmap="seismic", vmin=-64, vmax=63, interpolation="none")
plt.title("map of condition part separated by action")
plt.colorbar()
plt.savefig("exp_data\\seed" + str(seed) + "\\map of condition part separated by action")

plt.figure()
plt.imshow(afterNodesColored, cmap="gray", vmin=0, vmax=255, interpolation="none")
plt.title("map after learning coloerd by address bit")
plt.savefig("exp_data\\seed" + str(seed) + "\\map after learning coloerd by address bit")    

plt.show()

print("end of program")