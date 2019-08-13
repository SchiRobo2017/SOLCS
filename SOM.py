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
        self.nodes = np.random.rand(self.N*self.N,
                                    self.teachers.shape[1])
        self.ims = []

    def train(self):
        for i, teacher in enumerate(self.teachers):
            bmu = self._best_matching_unit(teacher)
            #d = np.linalg.norm(self.c - bmu, axis=1) #cとbmuの距離
            d = self._distance(self.c - bmu)
            L = self._learning_ratio(i)
            S = self._learning_radius(i, d)
            self.nodes += L * S[:, np.newaxis] * (teacher - self.nodes) #これを離散化する必要
            
            #進捗表示
            print("training iteration : "+ str(i)) 
            
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
        if axis==1:
            if ("#" in x): #futur warining
                # #含む距離
                return None
            return np.linalg.norm(x, axis=1, ord=1)
        else:
            return None

    def _best_matching_unit(self, teacher):
        #compute all norms (square)
        #norms = np.linalg.norm(self.nodes - teacher, axis=1)
        norms = self._distance(self.nodes - teacher) # #対応に向けて書き直し
        #bmu = np.argmin(norms) #normsを1次元にreshapeしたときのインデックス
        return np.unravel_index(np.argmin(norms),(self.N, self.N)) #N*N行列のargmax

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

N = 4
k = 2
bits = k + pow(2,k)
num_teachers = 10000   
#teachers = np.random.rand(10000, 3)
teachers = []
for i in range(num_teachers):
    teacher = []
    for j in range(bits):
        teacher.append(np.random.randint(2))
    teachers.append(teacher)
som = SOM(teachers, N=N, seed=10)

som.train()

m = som.nodes.reshape((N,N,bits))
m1 = np.round(m)

#fig = plt.figure()

# Initial map
#plt.imshow(som.nodes.reshape((N, N, 3)), interpolation='none')
#plt.show()

# Train
#som.train()

# Trained MAP
#plt.imshow(som.nodes.reshape((N, N, 3)), interpolation='none')
#plt.show()

#SOM.train作ったimsの中身をアニメーションで表示
#ani = animation.ArtistAnimation(fig, som.ims, interval=100)

#保存か表示か
#plt.show()
#ani.save("SOM.gif", writer="pillow")

#test variables
test = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]])

print("end of program")