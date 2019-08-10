# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 15:25:18 2019

@author: Nakata Koya
"""
import math
import random

class MUXProbGenerator:
    def __init__(self):
        self.k = 2
        self.length = int(self.k+math.pow(2, self.k)) #len of cl = k + 2^K
        self.N = 16 #問題数
        self.problems = []
    def generate(self):
        problem = []
        for i in range(self.length):
            if random.randrange(2)==0:
                problem.append(0)
            else:
                problem.append(1)
        addbit = problem[0:self.k]
        refbit = problem[self.k:]
        cal = ""
        #正解行動
        for x in range(len(addbit)):
            cal += str(addbit[x])
        ans = int(cal,2)
        self.ans = refbit[ans]
        problem.append(self.ans)
        print(problem)
        return problem
    def generateMultiProb(self):
        for i in range(self.N):
            self.problems.append(self.generate())
        return self.problems
            
mux = MUXProbGenerator()
mux.generateMultiProb()