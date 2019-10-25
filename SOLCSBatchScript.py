# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 16:00:02 2019

@author: Nakata Koya
"""

import SOM
import SOLCSFigureGenerator as fg

for i in range(10):
    print("seed", SOM.conf.seed_teacher, "started.")
    main = SOM.Main()
    main.main()
    fg.FigureGenerater().genFig(doesShow=False)
    SOM.conf.seed_teacher+=1