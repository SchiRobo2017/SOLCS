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
        if not seed is None:
            np.random.seed(seed)

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

def generateMUXNodes(k=2, num_teachers=10000, P_sharp = 0):
    teachers = []
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

def getGrayNodes(nodes, k=2):
    Max = k + k**2
    grayNodes = []
    for cl in nodes:        
        grayNodes.append(np.sum(cl)/Max)
    grayNodes = np.array(grayNodes)
    return grayNodes.reshape(N, N)

N = 100
k = 2
bits = k + pow(2,k)
num_teachers = 10000   
#teachers = np.random.rand(10000, 3)
teachers = generateMUXNodes()
som = SOM(teachers, N=N, seed=10)

m = som.nodes.reshape((N,N,bits)) #initial nodes of cl
m1 = np.round(m)
iniNodes = getGrayNodes(som.nodes)
iniAnsNodes = getAnsNodes(m1).reshape(N,N) #initial nodes of ansers

#plt.figure()
#plt.imshow(iniAnsNodes, cmap="gray", vmin=0, vmax=1, interpolation="none")
#plt.title("initial rounded")

plt.figure()
plt.imshow(iniNodes, cmap="gray", vmin=0, vmax=1, interpolation="none")
plt.title("initial condition part gray map")


som.train()


#ansNodes = getAnsNodes(np.round(som.nodes.reshape(N,N,bits))).reshape(N,N)
#plt.figure()
#plt.imshow(ansNodes, cmap="gray", vmin=0, vmax=1, interpolation="none")
#plt.title("after leaning")

afterNodes = getGrayNodes(np.round(som.nodes))
#afterNodes = getGrayNodes(som.nodes)
plt.figure()
plt.imshow(afterNodes, cmap="gray", vmin=0, vmax=1, interpolation="none")
plt.title("condition part gray map of after learning ")

plt.show()


#test variables
test = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]])

print("end of program")