# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 16:00:02 2019

@author: Nakata Koya
"""

#N=100でabout 40min ~ 1h10min/10000itr

import SOM
import SOLCSFigureGenerator as fg
import itertools

#複数バッチ
#teacher = list(range(1,6))
#train = list(range(10,16))

#1バッチ回すとき
teacher = list(range(1,2))
train = list(range(10,11))

seeds = itertools.product(teacher, train)

"""
for te, tr in seeds:
    SOM.conf.seed_teacher = te
    SOM.conf.seed_train = tr
    print("seed_teacher=", te, "seed_train=", tr)
    for adbit in SOM.conf.ADBIT_IDX.values():
        print("update_index:", str(adbit))
        SOM.conf.dirNameAdditionalStr = "_" + str(adbit) + "_N" + str(SOM.conf.N)
        SOM.conf.dirNameAdditionalStr += "_allbits_update"
        main = SOM.Main(upd_bit=adbit)
        main.main()
        fg.FigureGenerater(dirStr_result = SOM.conf.dirStr_result()).genFig(doesShow=False)
"""

#全ビット更新用
for te, tr in seeds:
    SOM.conf.seed_teacher = te
    SOM.conf.seed_train = tr
    print("seed_teacher=", te, "seed_train=", tr)
    print("update_index: all")
    SOM.conf.dirNameAdditionalStr = "_N" + str(SOM.conf.N)
    SOM.conf.dirNameAdditionalStr += "_allbits_update_501"
    main = SOM.Main()
    main.main()
    fg.FigureGenerater(dirStr_result = SOM.conf.dirStr_result()).genFig(doesShow=False)