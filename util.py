import pyabf
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from os import getcwd,listdir
import csv
import pandas as pd
import math


def analysis_Bias(abf,volt,single,timeStart=0, timeEnd = None, auto_fill = False):
    # 分析並抓出偏壓位置
    # single = (start_single,end_single)
    # 邏輯：
    # 1. 將偏壓按 （電壓-100）/2 分成兩類， 1為高電壓，0為低電壓（若電壓<100，則偏壓=1-偏壓，高低反轉）
    # 2. 若發生 1-->0事件（下降），該坐標為close，close_flag同時設定為false
    # 3. 若close_flag當下為false，該坐標則爲far，close_flag同時設定為true
        # Start_single: 若開始第一個點為far, 則close_flag初始值為True
    # 4. 
    abf.setSweep(0,channel=4)
    bias = np.digitize(abf.sweepY[timeStart:timeEnd],bins = [(volt-100)/2+100])
    if volt<100:
        bias = 1-bias
    close_index = []
    far_index = []
    index = []
    err_index = []
    debug = False #When debugging, set debug to True

    # Solving starting with a single far point problem
    if single[0]:
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
            
            # Automatic missing close/far handling
            if auto_fill:
                if len(far_index)==len(close_index):
                    ind_difs = np.array(far_index) - np.array(close_index)
                    if stepwise_outlier(ind_difs):
                        # Error occurs, either far or close missing
                        # Handling Strategy: 
                        # Goal: Insert estimate closed index
                        # (because either far or close missing, program will definitely assume far missing as close come first)
                        # but probably it happens because close didn't exists
                        # Now, last close index already be inserted, which should be far index

                        # Error Handling
                        print(f"Error point index:{len(close_index)-1}, Error value: {ind_difs[-1]}")
                        err_index.append(len(close_index)-1)
                        far_index[-1] = close_index[-1]
                        close_index[-1] = int(far_index[-1]-np.mean(ind_difs[:-1]))
                        close_index.append(i)
                        index[-1] = int(far_index[-1]-np.mean(ind_difs[:-1]))
                        index.append(i)
                        close_flag = False
                    
                    
    
    # Solve starting and/or ending with a single far point problem
    if single[0]:
        far_index = far_index[1:]
    if single[1]:
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
    return index,close_index,far_index,err_index

def findStable_Bias(abf,volt,single,mean_range=5,timeStart = 0, timeEnd = None,channel = 2, auto_fill = False):
    index,close_index,far_index,err_index = analysis_Bias(abf,volt,single,timeStart = timeStart,timeEnd = timeEnd, auto_fill = auto_fill)
    
    # mean_range default set as 5 because 1ms = 5 data points
    abf.setSweep(0,channel=channel)
    close = []
    far = []
    for ci in close_index:
        t = abf.sweepY[timeStart:timeEnd][ci-4:ci+1]
        mean = np.mean(np.array(t))
        close.append(mean)
        
    for fi in far_index:
        t = abf.sweepY[timeStart:timeEnd][fi-4:fi+1]
        mean = np.mean(np.array(t))
        far.append(mean)
        
    return np.array(close),np.array(far),close_index,far_index,err_index

# 定义递归展开函数
def flatten(lst):
    for item in lst:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item
            
def cal_points_qty(abf,index):
    x = []
    abf.setSweep(0,channel = 3)
    for c in index:
        x.append(abf.sweepY[c]) # Record X-axis value
    x = np.round(x,decimals = 0) # Round to bin
    newline = False
    lastx = -1
    matrix = []
    temp = []

    # Count the record qty according to X-axis
    for e in x:
        if lastx != e:
            if e>lastx:
                temp.append(1)
            else:
                matrix.append(temp)
                temp = [1]
        else:
            temp[-1]+=1
        lastx = e
    matrix.append(temp)
    return matrix
            
def print_points_qty(abf,index = None, matrix = None):
    # Either giving abf and index, or giving matrix
    if matrix is None:
        matrix = cal_points_qty(abf,index)
        
    for ind, row in enumerate(matrix):
        print("%3d."%ind,end=" ")
        for qty in row:
            print("%-2d"%qty,end=' ')
        print(f"Row Dimension:{len(row)} ; Row Total: {str(np.sum(flatten(row)))}")
    print("Total points: "+str(np.sum(flatten(matrix))))
    
def plot_wave(abf,volt,single = (False,False),timeStart = 0, timeEnd = None,channel = 2,auto_fill = False):
    abf.setSweep(0,channel = channel)
    plt.figure(figsize=(18,5))
    plt.plot(abf.sweepX[timeStart:timeEnd],abf.sweepY[timeStart:timeEnd],color = 'green')
    
    #plot the stable 
    index,close_index,far_index,_ = analysis_Bias(abf,volt,single,timeStart = timeStart,timeEnd = timeEnd,auto_fill = auto_fill)
    abf.setSweep(0,channel = channel)
    barX_start = []
    barY_start = []
    barX_end = []
    barY_end = []
    counting = 0
    for i in close_index:
        length = 21
        counting+=1
        for j in range(length):
            barX_end.append(abf.sweepX[timeStart:timeEnd][i])
            barY_end.append(abf.sweepY[timeStart:timeEnd][i]+(j-int(length/2)+1))
            barX_start.append(abf.sweepX[timeStart:timeEnd][i-4])
            barY_start.append(abf.sweepY[timeStart:timeEnd][i-4]+(j-int(length/2)+1))
        plt.annotate(str(counting),xy=(abf.sweepX[timeStart:timeEnd][i-4],abf.sweepY[timeStart:timeEnd][i-4]+(j-int(length/2)+1)+1))
    plt.scatter(barX_end,barY_end,s=1,facecolors='b', edgecolors='b')
    plt.scatter(barX_start,barY_start,s=1,facecolors='r',edgecolors='r')
    
    barX_start = []
    barY_start = []
    barX_end = []
    barY_end = []
    counting = 0
    for i in far_index:
        length = 21
        counting+=1
        for j in range(length):
            barX_end.append(abf.sweepX[timeStart:timeEnd][i])
            barY_end.append(abf.sweepY[timeStart:timeEnd][i]+(j-int(length/2)+1))
            barX_start.append(abf.sweepX[timeStart:timeEnd][i-4])
            barY_start.append(abf.sweepY[timeStart:timeEnd][i-4]+(j-int(length/2)+1))
        plt.annotate(str(counting),xy=(abf.sweepX[timeStart:timeEnd][i-4],abf.sweepY[timeStart:timeEnd][i-4]+(j-int(length/2)+1)+1))
    plt.scatter(barX_end,barY_end,s=1,facecolors='m', edgecolors='m')
    plt.scatter(barX_start,barY_start,s=1,facecolors='c',edgecolors='c')
    plt.ylabel(abf.sweepLabelY)
    plt.xlabel(abf.sweepLabelX)
    plt.show()
    return len(close_index),len(far_index)

def plot_wave_with_index(abf,close_index,far_index,timeStart = 0, timeEnd = None,channel = 2):
    abf.setSweep(0,channel = channel)
    plt.figure(figsize=(18,5))
    plt.plot(abf.sweepX[timeStart:timeEnd],abf.sweepY[timeStart:timeEnd],color = 'green')
    
    #plot the stable
    close_index = [x-timeStart for x in close_index if timeStart <= x <= timeEnd]
    far_index = [x-timeStart for x in far_index if timeStart <= x <= timeEnd]
    abf.setSweep(0,channel = channel)
    barX_start = []
    barY_start = []
    barX_end = []
    barY_end = []
    counting = 0
    for i in close_index:
        length = 21
        counting+=1
        for j in range(length):
            barX_end.append(abf.sweepX[timeStart:timeEnd][i])
            barY_end.append(abf.sweepY[timeStart:timeEnd][i]+(j-int(length/2)+1))
            barX_start.append(abf.sweepX[timeStart:timeEnd][i-4])
            barY_start.append(abf.sweepY[timeStart:timeEnd][i-4]+(j-int(length/2)+1))
        plt.annotate(str(counting),xy=(abf.sweepX[timeStart:timeEnd][i-4],abf.sweepY[timeStart:timeEnd][i-4]+(j-int(length/2)+1)+1))
    plt.scatter(barX_end,barY_end,s=1,facecolors='b', edgecolors='b')
    plt.scatter(barX_start,barY_start,s=1,facecolors='r',edgecolors='r')
    
    barX_start = []
    barY_start = []
    barX_end = []
    barY_end = []
    counting = 0
    for i in far_index:
        length = 21
        counting+=1
        for j in range(length):
            barX_end.append(abf.sweepX[timeStart:timeEnd][i])
            barY_end.append(abf.sweepY[timeStart:timeEnd][i]+(j-int(length/2)+1))
            barX_start.append(abf.sweepX[timeStart:timeEnd][i-4])
            barY_start.append(abf.sweepY[timeStart:timeEnd][i-4]+(j-int(length/2)+1))
        plt.annotate(str(counting),xy=(abf.sweepX[timeStart:timeEnd][i-4],abf.sweepY[timeStart:timeEnd][i-4]+(j-int(length/2)+1)+1))
    plt.scatter(barX_end,barY_end,s=1,facecolors='m', edgecolors='m')
    plt.scatter(barX_start,barY_start,s=1,facecolors='c',edgecolors='c')
    plt.ylabel(abf.sweepLabelY)
    plt.xlabel(abf.sweepLabelX)
    plt.show()
    return len(close_index),len(far_index)


def plot_colormap(data,title,path = "",vmin = None, vmax = None,figsize = None):
    """
    Helper function to plot data with associated colormap.
    """
   # fig, axs = plt.subplots(1, n, figsize=(n * 2 + 2, 3),
    #                        constrained_layout=True, squeeze=False)
    if figsize is None:
        figure, axes = plt.subplots(figsize=(data.shape[1]/3,data.shape[0]/10))
    else:
        figure, axes = plt.subplots(figsize=figsize)
    if vmin is None:
        vmin = min(data.flatten())
    if vmax is None:
        vmax= max(data.flatten())
    psm = axes.pcolormesh(data, cmap='rainbow',rasterized=True,vmin=vmin, vmax=vmax)
    figure.colorbar(psm, ax=axes)
    axes.invert_yaxis()
    if abs(data.shape[0]-data.shape[0])<=1:
        #print(data.shape[1],data.shape[0])
        axes.set_aspect('equal', adjustable='box')
    plt.title(title + f" \n({vmin:.4f},{vmax:.4f})")
    
    mean = np.mean(data.flatten())
    std = np.std(data.flatten())
    plt.figtext(0.5, -0.05, f"%lf ± %lf" %(mean,std), 
                ha="center", fontsize=10, color="black")
    plt.savefig(path+" "+title)
    plt.show()
    
def process_end_ignore(matrix, ignore):
    # matrix: a 2-d list
    '''
    ## Start ignore:
    ignore_num = ignore[0]
    for row in range(len(matrix)):
        for col in range(len(matrix[row])):
            if col>=ignore_num:
                matrix[row][col] -= ignore_num
                row = len(matrix)
                break
            else:
                ignore_num -= matrix[row][col]
                matrix[row][col] = 0
    '''
    ## End ignore
    ignore_num = ignore[1]
    for row in reversed(range(len(matrix))):
        for col in reversed(range(len(matrix[row]))):
            if matrix[row][col] >= ignore_num:
                matrix[row][col] -= ignore_num
                matrix = [row for row in matrix if not all(x == 0 for x in row)]
                return matrix
            else:
                ignore_num -= matrix[row][col]
                matrix[row][col] = 0
    matrix = [row for row in matrix if not all(x == 0 for x in row)]
    return matrix

def capture(data, first_row_repeat, ignore, extra, dim):
    # Capture Data
    # data: 1-d list, before remove ignore and extra points
    # first_row_repeat: each point in first row has repeated how many times, not consider start_ignore
    # ignore: (start_ignore, end_ignore)
    # extra: (start_extra, end_extra)
    # dim: (Row, Col)
    # According to requirements, first row is ignored
    
    data = data[ignore[0]:-ignore[1]]
    data = data[first_row_repeat*dim[1]:]
    data = np.array(data).reshape((dim[0],dim[1]+extra[0]+extra[1]))
    data_temp = []
    for row in range(len(data)):
        if extra[1]==0:
            data_temp.append(data[row][extra[0]:])
        else:
            data_temp.append(data[row][extra[0]:-extra[1]])
    data_temp = np.array(data_temp)
    return data_temp
    
def write_target(target, file_path):
    # 计算每列的平均值和标准差
    mean = np.mean(target)
    std = np.std(target)

    stats = [['Mean',mean],['Standard Deviation',std]]

    # 将 `target` 和 `stats` 输出到 CSV 文件，不包含列名
    target_file_path = "result\\"+ file_path + '_data.csv'
    stats_file_path =  "stats\\"+file_path + '_stats.csv'
    
    with open(target_file_path, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        # write the data
        if len(target.shape)==2:
            for x in target:
                writer.writerow(x)
                
        if len(target.shape)==1:
            writer.writerow(target)
            
    with open(stats_file_path, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        # write the data
        for x in stats:
            writer.writerow(x)
            
# Progressively check which diffenrence may be error
def stepwise_outlier(arr,tol = 30):
    # tol: accept inside the boundary (mean+- tol*std)
    # Only check the last one
    mean = np.mean(arr)
    std = np.std(arr)
    upper_bound = mean+tol*std
    lower_bound = mean-tol*std
    if arr[-1]> upper_bound or  arr[-1]< lower_bound:
        print(f"Now:{arr[-1]:.2f},Mean: {mean:.2f}, Std:{std:.2f},Limits:[{lower_bound:.2f},{upper_bound:.2f}]")
        return True
    else:
        
        return False
    
def map_ind_coor(index, Row, Col, ignore, extra, first_row_repeat):
    # Case 1: Inside start ignore bounds
    if index < ignore[0]:
        print(f"Index {index} is located among start ignore points")
        return (0,index)
    # Case 2: At first row
    elif index < (ignore[0]+Col*first_row_repeat):
        col = index-ignore[0]
        col = math.floor(col/first_row_repeat)
        print(f"Index {index} is located at Row = 1, Col = {col}")
        return (1,col)
    # Case 3: Among Row 2~Last row exclude end ignore
    elif index<((Row-1)*(Col+extra[0]+extra[1])+ignore[0]+Col*first_row_repeat):
        row = index - (ignore[0]+Col*first_row_repeat)
        row = math.floor(row/(Col+extra[0]+extra[0]))+1
        
        col = index - (ignore[0]+Col*first_row_repeat)
        col = col-(row-1)*(Col+extra[0]+extra[1])
        if col <= extra[0]:
            print(f"Index {index} is located at Row = {row}, among start extra points.")
        elif col < Col+extra[0]:
            print(f"Index {index} is located at Row = {row}, Col = {col-extra[0]}")
        else:
            print(f"Index {index} is located at Row = {row}, among end extra points.")
        return (row,col)
    else:
        print("Index is located among end ignore points")
        return (Row,-ignore[1])

def check_close_fair_pair(close_ind,far_ind, Row, Col, ignore, extra, first_row_repeat, volt, abf):
    difs = np.array(far_ind) - np.array(close_ind)
    plt.hist(difs, bins=10, alpha=0.7, edgecolor='black')  # You can change the number of bins as needed
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of the Array')
    plt.show()

    err_ind = []
    for i in range(len(difs)):
        if stepwise_outlier(difs[:i+1]):
            err_ind.append(i)

    # Check corresponding wave
    for err in err_ind:
        print(f"Error point index:{err}, Error value: {difs[err]}")
        map_ind_coor(err, Row, Col, ignore, extra, first_row_repeat)
        timeStart = close_ind[err]-20000
        timeEnd = far_ind[err]+20000
        plot_wave(abf,volt,timeStart = timeStart, timeEnd = timeEnd,channel=2)
        plot_wave(abf,volt,timeStart = timeStart, timeEnd = timeEnd,channel=4)