import pyabf
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from os import getcwd,listdir
import csv


def analysis_Bias(abf,volt,start_single,end_single,timeStart=0, timeEnd = None):
    # 分析並抓出偏壓位置
    # 邏輯：
    # 1. 將偏壓按 （電壓-100）/2 分成兩類， 1為高電壓，0為低電壓（若電壓<100，則偏壓=1-偏壓，高低反轉）
    # 2. 若發生 1-->0事件（下降），該坐標為close，close_flag同時設定為false
    # 3. 若close_flag當下為false，該坐標則爲far，close_flag同時設定為true
        # Start_single: 若開始第一個點為far, 則close_flag初始值為True
    # 4. 
    abf.setSweep(0,channel=4)
    bias = np.digitize(abf.sweepY[timeStart:timeEnd],bins = [(volt-100)/2])
    if volt<100:
        bias = 1-bias
    close_index = []
    far_index = []
    index = []
    debug = False #When debugging, set debug to True

    # Solving starting with a single far point problem
    if start_single:
        close_flag = False
    else:
        close_flag = True
        
    for i in range(0,len(bias)-1):
        dif = bias[i+1]-bias[i]
        
        #只需考慮bias差值小於-1的部分（下降點）
        #另外應該從volt點下降，10是寬限，比方説800mv，則下降應從於700mv
        '''
        if dif<-1 and bias[i]>volt-95 and bias[i+1]<volt-95: 
            f = False
            for gap in range(5):
                if i+gap < len(bias):
                    if bias[i+gap]<volt-95:
                        f = True
                        break
            if f == False:
                break
            '''
        if dif == -1:
            index.append(i)
            if close_flag:
                close_index.append(i)
                close_flag = False
            else:
                far_index.append(i)
                close_flag = True
    
    # Solve starting and/or ending with a single far point problem
    if start_single:
        far_index = far_index[1:]
    if end_single:
        close_index = close_index[:-1]
    
    print("Close point found: "+str(len(close_index)))
    print("Far point found: "+str(len(far_index)))
    
    if debug:
        print("Close point are:")
        for i in close_index:
            print(abf.sweepX[timeStart:timeEnd][i],bias[i])

        print("Far point are:")
        for i in far_index:
            print(abf.sweepX[timeStart:timeEnd][i],bias[i])
    return index,close_index,far_index