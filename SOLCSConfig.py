# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 06:44:52 2019

@author: Nakata Koya
"""
import datetime

#caution: 中間発表の結果はteacher=10, seedはSOM初期梶にNone渡し
seed_teacher = 10 #入力データseed
seed_train = 10 #マップ初期化シード
N = 100
head = 3
k = 2
includeAns = True
bits = k + 2**k
if includeAns==True:
    bits+=1
num_teachers = 10000 #default=10000 収束する   
#dirStr_result = "exp_data\\seed" + str(seed_teacher) #seedが途中で変わったときに対応できない

def dirStr_result(): #seedの動的変更に対応
    return "exp_data\\seed" + str(seed_teacher)

#dt_now = datetime.datetime.now().strftime('%Y%m%d%H%M%S') #モジュールをロードした時刻
def dt_now(): #この関数を呼び出した時刻
    return datetime.datetime.now().strftime('%Y%m%d%H%M%S')