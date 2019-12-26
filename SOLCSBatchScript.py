# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 16:00:02 2019

@author: Nakata Koya
"""

import SOM
import SOLCSFigureGenerator as fg
import itertools

#teacher = list(range(1,6))
#train = list(range(10,16))
teacher = list(range(1,2))
train = list(range(10,11))
seeds = itertools.product(teacher, train)

for te, tr in seeds:
    SOM.conf.seed_teacher = te
    SOM.conf.seed_train = tr
    print("seed_teacher=", te, "seed_train=", tr)
    for adbit in SOM.conf.ADBIT_IDX.values():
        print("update_index:", str(adbit))
        main = SOM.Main(upd_bit=adbit)
        main.main()
        fg.FigureGenerater(dirStr_result = SOM.conf.dirStr_result().genFig(doesShow=False))
