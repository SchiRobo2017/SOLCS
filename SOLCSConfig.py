# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 06:44:52 2019

@author: Nakata Koya
"""
import datetime

#実験データディレクトリ名にに追加するタグ
dirNameAdditionalStr = "_debug"
#caution: 中間発表の結果はteacher=10, trainはSOM初期化時にNone渡し
seed_teacher = 10 #入力データseed
seed_train = None #マップ初期化シード
N = 100
head = 3
k = 2
includeAns = True
doesErrorCorrect = False
#生成する問題に答えを付与するか
includeRewards = False
bits = k + 2**k
if includeAns==True:
    bits+=1
num_teachers = 10000 #default=10000 収束する   
#dirStr_result = "exp_data\\seed" + str(seed_teacher) #seedが途中で変わったときに対応できない

def dirStr_result(): #seedの動的変更に対応
    dirStr = "exp_data\\teacher" + str(seed_teacher) + "_train" +str(seed_train)
    if doesErrorCorrect == True:
        return dirStr + "_error_corrected"
    return dirStr + dirNameAdditionalStr

#dt_now = datetime.datetime.now().strftime('%Y%m%d%H%M%S') #モジュールをロードした時刻
def dt_now(): #この関数を呼び出した時刻
    return datetime.datetime.now().strftime('%Y%m%d%H%M%S')