import pyabf
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from os import getcwd,listdir
import csv


def analysis_Bias(abf,volt,single,timeStart=0, timeEnd = None):
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
    return index,close_index,far_index

def findStable_Bias(abf,volt,single,mean_range=5,timeStart = 0, timeEnd = None,channel = 2):
    index,close_index,far_index = analysis_Bias(abf,volt,single,timeStart = timeStart,timeEnd = timeEnd)
    
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
        
    return np.array(close),np.array(far),close_index,far_index

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
            
def print_points_qty(abf = None,index = None, matrix = None):
    # Either giving abf and index, or giving matrix
    if matrix is None:
        matrix = calculate_points_qty(abf,index)
        
    for ind, row in enumerate(matrix):
        print("%3d."%ind,end=" ")
        for qty in row:
            print("%-2d"%qty,end=' ')
        print(f"Row Dimension:{len(row)} ; Row Total: {str(np.sum(flatten(row)))}")
    print("Total points: "+str(np.sum(flatten(matrix))))
    
def plot_wave(abf,volt,start_single=False,end_single=False,timeStart = 0, timeEnd = None,channel = 2):
    abf.setSweep(0,channel = channel)
    plt.figure(figsize=(18,5))
    plt.plot(abf.sweepX[timeStart:timeEnd],abf.sweepY[timeStart:timeEnd],color = 'green')
    
    #plot the stable 
    index,close_index,far_index = analysis_Bias(abf,volt,start_single,end_single,timeStart = timeStart,timeEnd = timeEnd)
    abf.setSweep(0,channel = channel)
    barX_start = []
    barY_start = []
    barX_end = []
    barY_end = []
    counting = 0
    for i in index:
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
    plt.ylabel(abf.sweepLabelY)
    plt.xlabel(abf.sweepLabelX)
    plt.show()
    return len(close_index),len(far_index)


def plot_colormap(data,title,path = "",vmin = 0.98, vmax = 1.20,figsize = None):
    """
    Helper function to plot data with associated colormap.
    """
   # fig, axs = plt.subplots(1, n, figsize=(n * 2 + 2, 3),
    #                        constrained_layout=True, squeeze=False)
    if figsize is None:
        figure, axes = plt.subplots(figsize=(data.shape[1]/3,data.shape[0]/10))
    else:
        figure, axes = plt.subplots(figsize=figsize)
    psm = axes.pcolormesh(data, cmap='rainbow',rasterized=True,vmin=min(data.flatten()), vmax=max(data.flatten()))
    figure.colorbar(psm, ax=axes)
    axes.invert_yaxis()
    if abs(data.shape[0]-data.shape[0])<=1:
        #print(data.shape[1],data.shape[0])
        axes.set_aspect('equal', adjustable='box')
    plt.title(title)
    
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

def calculate_mean(data, matrix, ignore, extra, dim):
    # Calculate mean of data at each point, points qty is recorded in matrix
    # Matrix: m*n list, before remove ignore and extra points
    # ignore: (start_ignore, end_ignore)
    # extra: (start_extra, end_extra)
    # dim: (Row, Col)
    # According to requirements, first row is ignored

    matrix = process_end_ignore(matrix,ignore)
    
    p_ind = 0
    for row in range(len(matrix)):
        for col in range(len(matrix[row])):
            #print("Points considered: "+str(matrix[row][col])+", "+str(p_ind)+"-"+str(p_ind+matrix[row][col]))
            temp = np.zeros(matrix[row][col])
            #print("\t Before extract: ",end='')
            #print(temp)
            for ind,i in enumerate(range(p_ind,p_ind+matrix[row][col])):
                temp[ind] = data[i]
            #print("\t After extract: ",end='')
            #print(temp)
            p_ind += matrix[row][col]
            if col == 0:
                matrix[row][col] = np.mean(temp[extra[0]:])
            elif col == len(matrix[row])-1 and extra[1] != 0:
                matrix[row][col] = np.mean(temp[:-extra[1]])
            else:
                matrix[row][col] = np.mean(temp)
            
    # Capture required dimension(Row), ignore first row
    matrix = matrix[1:1+dim[0]][:]
    matrix = np.array(matrix).reshape(dim)
    return matrix
    
    