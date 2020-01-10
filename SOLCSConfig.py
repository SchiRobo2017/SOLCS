# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 06:44:52 2019

@author: Nakata Koya
"""
import datetime

#実験データディレクトリ名にに追加するタグ
dirNameAdditionalStr = ""
#caution: 中間発表の結果はteacher=10, trainはSOM初期化時にNone渡し
seed_teacher = 10 #入力データseed
seed_train = None #マップ初期化シード
N = 5 #default=100
head = 3
k = 2
includeAns = True
doesErrorCorrect = False
#生成する問題に報酬を付与するか
includeRewards = False
bits = k + 2**k
if includeAns==True:
    bits+=1
num_teachers = 10001 #default=10000   
#dirStr_result = "exp_data\\seed" + str(seed_teacher) #seedが途中で変わったときに対応できない

#SOMの更新部に関する定義
ADBIT00 = [0,1,2]
ADBIT01 = [0,1,3]
ADBIT10 = [0,1,4]
ADBIT11 = [0,1,5]
ADBIT_IDX = {"BLACK":[0,1,2], "RED":[0,1,3], "GREEN":[0,1,4], "BLUE":[0,1,5]}

ADBIT_VALS = [[0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1]]
ACTIONS = [0,1]

dt_loaded = datetime.datetime.now().strftime('%Y%m%d%H%M%S') #モジュールをロードした時刻

def dirStr_result(): #seedの動的変更に対応
    dirStr = "exp_data\\teacher" + str(seed_teacher) + "_train" +str(seed_train)
    #if doesErrorCorrect == True:
    #    dirStr += "_error_corrected"
    dirStr += dirNameAdditionalStr
    dirStr += "_" + str(dt_loaded)
    return dirStr

def dt_now(): #この関数を呼び出した時刻
    return datetime.datetime.now().strftime('%Y%m%d%H%M%S')