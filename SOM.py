# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 15:36:50 2019

@author: Nakata Koya
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class SOM():

    def __init__(self, teachers, N, seed=None):
        self.teachers = np.array(teachers)
        self.n_teacher = self.teachers.shape[0]
        self.N = N
        #if not seed is None:
        #    np.random.seed(seed)

        x, y = np.meshgrid(range(self.N), range(self.N)) #格子点の生成
        self.c = np.hstack((y.flatten()[:, np.newaxis],
                            x.flatten()[:, np.newaxis])) #座標の配列に変換
        self.nodes = np.round(np.random.rand(self.N*self.N,
                                    self.teachers.shape[1]))
        self.ims = []

    def train(self):
        for i, teacher in enumerate(self.teachers):
            bmu = self._best_matching_unit(teacher)
            d = np.linalg.norm(self.c - bmu, axis=1) #cとbmuの距離
            #d = self._distance(self.c - bmu)
            L = self._learning_ratio(i)
            S = self._learning_radius(i, d)
            self.nodes += L * S[:, np.newaxis] * (teacher - self.nodes) #これを離散化する必要
            
            #進捗表示
            #print("training iteration : "+ str(i)) 
            
            #適当なインターバルで現在の学習状態をimsに格納
            """
            if i%30 == 0:
                im = plt.imshow(self.nodes.reshape((N, N, 3)), interpolation='none')
                self.ims.append([im])
            """
            
            #plt.cla()
            #plt.imshow(self.nodes.reshape((N, N, 3)))
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
        """
        dist_array = []
        
        for row in x:
            dist_array.append(calc_dist(row))
        
        dist_array = np.array(dist_array)
        return dist_array.reshape(self.N, self.N, 1)
        """
        
    def _best_matching_unit(self, teacher):
        norms = np.linalg.norm(self.nodes - teacher, axis=1)
        #norms = self._distance(self.nodes - teacher)
        #print(np.argmin(norms))
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

def generateMUXNodes(k=2, num_teachers=10000, seed = None, P_sharp = 0):
    teachers = []
    bits = k + pow(2,k)
    for i in range(num_teachers):
        teacher = []
        for j in range(bits):
            if np.random.rand() < P_sharp:
                teacher.append("#")
            else:
                teacher.append(np.random.randint(2))
        teachers.append(teacher)
    return teachers

def getAnsNodes(nodes, k=2):
    ansNodes = []
    for row in nodes:
        for elm in row:
            addbit = elm[0:k]
            refbit = elm[k:]
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

def getColoredNodes(nodes, k=2, scale="k-scale", color="gray"):
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
        
    if scale == "k-scale":
        for cl in nodes:        
            coloredNodes.append(np.sum(cl)/Max)
    elif scale=="63":
        for cl in nodes:
            cljoined = [str(int(i)) for i in cl]
            cljoined = "".join(cljoined)
            clIntScale = int(cljoined,2)
            coloredNodes.append(clIntScale)
            #coloredNodes.append(int("".join([str(int(i)) for i in cl]))) #一行で書けばこう
    else:
        raise ValueError("scaleに渡す引数が間違ってるよ")

    coloredNodes = np.array(coloredNodes, dtype = np.uint8)
    return coloredNodes.reshape(N, N)

seed = 10
N = 100
k = 2
bits = k + pow(2,k)
num_teachers = 10000 #default=10000 収束する   
#teachers = np.random.rand(10000, 3)
teachers = generateMUXNodes()
np.random.seed(seed)
som = SOM(teachers, N=N, seed=seed)

m = som.nodes.reshape((N,N,bits)) #initial nodes of cl
m1 = np.round(m)
iniNodes = getColoredNodes(som.nodes, scale="63")
iniNodesColored = getColoredNodes(np.round(som.nodes), color="colored")
iniAnsNodes = getAnsNodes(m1).reshape(N,N) #initial nodes of ansers


"""
plt.figure()
plt.imshow(iniAnsNodes, cmap="gray", vmin=0, vmax=1, interpolation="none")
plt.title("initial map of actions")
"""

"""
plt.figure()
plt.imshow(iniNodes, cmap="gray", vmin=0, vmax=63, interpolation="none")
plt.title("initial map of condition by 64 gray scale")
"""

"""
plt.figure()
plt.imshow(iniNodesColored, cmap="gray", vmin=0, vmax=255, interpolation="none")
plt.title("initial map colored by address bit")
"""

print("traing has started.")
som.train()
print("training has finished.")


ansNodes = getAnsNodes(np.round(som.nodes.reshape(N,N,bits))).reshape(N,N)
afterNodes = getColoredNodes(som.nodes, scale="63")
afterNodesRounded = getColoredNodes(np.round(som.nodes), scale="63") #丸めると不思議な模様が！
afterNodesColored = getColoredNodes(np.round(som.nodes), color="colored")

plt.figure()
plt.imshow(ansNodes, cmap="gray", vmin=0, vmax=1, interpolation="none")
plt.title("map of condition part after leaning")

plt.figure()
plt.imshow(afterNodesRounded, cmap="gray", vmin=0, vmax=63, interpolation="none")
plt.title("map of rounded action part after learning")

plt.figure()
plt.imshow(afterNodesColored, cmap="gray", vmin=0, vmax=255, interpolation="none")
plt.title("map after learning coloerd by address bit")

plt.show()


#test variables
test = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]])

print("end of program")