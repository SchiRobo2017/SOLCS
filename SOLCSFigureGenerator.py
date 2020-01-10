# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 11:24:13 2019

@author: Nakata Koya
"""
import pickle
import math
import numpy as np
import matplotlib.pyplot as plt
import SOM

class FigureGenerater():
    def __init__(self, dirStr_result):
        self.dirStr_result = dirStr_result
        with open(self.dirStr_result +  "\\nodes.bin", "rb") as nodes:
            self.nodes = pickle.load(nodes)        
        
        #todo
        self.actNodesRealNum = self.nodes[:,-1].reshape(SOM.conf.N, SOM.conf.N)
        self.actNodes = np.round(self.actNodesRealNum)
        self.correctActNodes = SOM.getAnsNodes(np.round(self.nodes)).reshape(SOM.conf.N, SOM.conf.N)
        self.afterNodesRounded_hamming = getColoredNodes(np.round(self.nodes),
                                        color="bits-scale")
        self.afterNodesRounded = getColoredNodes(np.round(self.nodes),
                                        color="bits2decimal-scale")
    
        #todo
        self.afterNodesReverse = np.round(self.nodes)[:,0:-1] #get 6bit nodes
        #todo
        self.afterNodesReverse = getColoredNodes(self.afterNodesReverse[:,::-1], color="bits2decimal-scale")
    
        self.afterNodesSeparated = self.afterNodesRounded.copy()
        self.afterNodesColored = getColoredNodes(np.round(self.nodes), color="colored")        
    
    def genFig(self, doesShow=True):
        """
        Showing map
        """
        dt_now = SOM.conf.dt_now()
        
        """
        #plt.figure()
        #plt.imshow(iniCorrectActNodes, cmap="gray", vmin=0, vmax=1, interpolation="none")
        #plt.title("initial map of actions")
        #plt.savefig(dirStr_result + "\\initial map of actions.png")
        
        #plt.figure()
        #plt.imshow(iniNodes, cmap="gray", vmin=0, vmax=63, interpolation="none")
        #plt.title("initial map of condition by 64 scale")
        #plt.savefig(dirStr_result + str(seed) + "\\initial map of condition by 64 scale")
        
        #plt.figure()
        #plt.imshow(iniNodesColored, cmap="gray", vmin=0, vmax=255, interpolation="none")
        #plt.title("initial map colored by address bit")
        #plt.savefig(dirStr_result + "\\initial map colored by address bit")
        """
        
        """
        plt.figure()
        plt.imshow(actNodesR, cmap="gray", vmin=0, vmax=1,
                   interpolation="none")
        plt.title("map of action part after leaning(continuous value)")
        plt.savefig(dirStr_result + 
                    "\\map of action part after leaning(countinuous value)")
        """
        
        plt.figure()
        plt.imshow(self.actNodes, cmap="gray", vmin=0, vmax=1,
                   interpolation="none")
        plt.title("map of action part after leaning")
        plt.savefig(self.dirStr_result +
                    "\\map of action part after leaning" 
                    + dt_now)
        
        plt.figure()
        plt.imshow(self.correctActNodes, cmap="gray", vmin=0, vmax=1, interpolation="none")
        plt.title("map of correct action part after leaning")
        plt.savefig(self.dirStr_result +
                    "\\map of correct action part after leaning"
                    + dt_now)
        
        plt.figure()
        plt.imshow(self.afterNodesRounded, cmap="gray", vmin=0, vmax=63, interpolation="none")
        plt.title("map of condition part after learning")
        plt.colorbar()
        plt.savefig(self.dirStr_result +
                    "\\map of condition part after learning"
                    + dt_now)
        
        plt.figure()
        plt.imshow(self.afterNodesColored, cmap="gray", vmin=0, vmax=255, interpolation="none")
        plt.title("map after learning coloerd by address and act")
        plt.savefig(self.dirStr_result +
                    "\\map after learning coloerd by address and act" 
                    + dt_now)
    
        """
        #全分類子のマッピング
        main.som.head = None
        
        #mappingDataList = np.array(generateMUXNodes(100,includeAns=True))
        mappingDataSequencial = []
        for i in range(0,64):
            mappingDataSequencial.append(str(bin(i))[2:].zfill(6))
            
        mappingDataSequencial = [list(x) for x in mappingDataSequencial]
        for i, row in enumerate(mappingDataSequencial):
            mappingDataSequencial[i] = [int(x) for x in row]
        
        for cl in mappingDataSequencial:
            cl.append(getAns(cl))
        
        mappingDataSequencial = np.array(mappingDataSequencial).reshape(len(mappingDataSequencial),len(mappingDataSequencial[0]))
        """

        """        
        #実数ノードに全入力を曝露したデータ
        nodes_mapped_new_input = main.som.mapping(mappingDataSequencial)
        
        nodes_mapped_new_input_colored = getColoredNodes(nodes_mapped_new_input, color="colored")
    
        plt.figure()
        plt.imshow(nodes_mapped_new_input_colored, cmap="gray", vmin=0, vmax=255, interpolation="none")
        plt.title("map of new classifier input")
        plt.savefig(dirStr_result +
                    "\\map of new classifier input" 
                    + dt_now)
        """   
   
  
        #正解 -> 0
        #誤り:正解が1で獲得が0　-> 1
        #誤り:正解が0で獲得が1 -> -1
        #となるノードを取得
        nodes_incorrect = self.actNodes - self.correctActNodes
        
        nodes_tmp = np.round(self.nodes)
        
        #hack
        for i, val in enumerate(nodes_incorrect.reshape(SOM.conf.N * SOM.conf.N)):
            if val == 0:
                #white
                nodes_tmp[i] = [-1,-1,-1,-1,-1,-1,-1]
            elif val == 1:
                continue
            elif val == -1:
                continue
            else:
                print("reached else block")

        plt.figure()
        plt.imshow(getColoredNodes(nodes_tmp, color="colored") , cmap="gray", vmin=-1, vmax=1, interpolation="none")
        plt.title("map of incorrect classifier")
        plt.savefig(self.dirStr_result +
                    "\\map of incorrect classifier" 
                    + dt_now)

        if doesShow == True:
            plt.show()
            
        print(self.dirStr_result, "saved")
        
def getColoredNodes(nodes, k=2, color="gray"): #nodes.shape must be [N*N, bits]
    Max = k + k**2
    N = int(math.sqrt(nodes.shape[0])) #edge length of the map
    coloredNodes = []
    
    #アドレスビットと行動で色分け
    if color=="colored":
        for cl in nodes:
            addBits = None
            ansBit = None
            if cl[0] != -1:
                addBitsArray = cl[:k]
                #refBitsArray = cl[k:-1]
                addBits = [str(int(i)) for i in addBitsArray]
                addBits = "".join(addBits)
                #ansBit = refBitsArray[int(addBits,2)] #正解行動
                #todo
                ansBit = cl[-1] #SOMが獲得した正解

            if addBits=="00": #黒
                if ansBit == 1:
                    coloredNodes.append([0,0,0])
                else:
                    coloredNodes.append([128,128,128])    
            elif addBits=="01": #R
                if ansBit == 1:
                    coloredNodes.append([128,0,0])
                else:
                    coloredNodes.append([255,0,0])
            elif addBits=="10": #G
                if ansBit == 1:
                    coloredNodes.append([0,128,0])
                else:
                    coloredNodes.append([0,255,0])
            elif addBits=="11": #B
                if ansBit == 1:
                    coloredNodes.append([0,0,128])
                else:
                    coloredNodes.append([0,0,255])
            else: #W
                coloredNodes.append([255,255,255])

        coloredNodes = np.array(coloredNodes, dtype = np.uint8)
        return coloredNodes.reshape(N, N, 3)
    
    elif color == "bits-scale":
        for cl in nodes:        
            coloredNodes.append(np.sum(cl[:-1])/Max)
            
        coloredNodes = np.array(coloredNodes)
        return coloredNodes.reshape(N, N)

    elif color == "bits2decimal-scale":
        for cl in nodes:
            cljoined = [str(int(i)) for i in cl]
            cljoined = cljoined[:k+k**2] #act bitを除外
            cljoined = "".join(cljoined)
            clIntScale = int(cljoined,2) #2進数の2
            coloredNodes.append(clIntScale)
            #coloredNodes.append(int("".join([str(int(i)) for i in cl]))) #一行で書けばこう
            
        #coloredNodes = np.array(coloredNodes, dtype = np.uint8)
        coloredNodes = np.array(coloredNodes)
        return coloredNodes.reshape(N, N)
    else:
        raise ValueError("colorに渡す引数が間違ってるよ")

    raise ValueError("colorに渡す引数が間違ってるよ")
        
if __name__ == "__main__":
    #FigureGenerater(default = seed10).genFig()
    FigureGenerater("exp_data\\teacher10_trainNone_debug").genFig() #hack: magic number