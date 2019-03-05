import numpy as np
import pandas as pd
import copy
from matplotlib.figure import Figure
#from matplotlib import pyplot as plt

#colours = ["b", "m", "r"]

def update_line(axVar, xdata, ydata):
    axVar.set_xdata(xdata)
    axVar.set_ydata(ydata)
    return axVar

#Returns a matplotlib subplot to be displayed       
def draw_subplot(path,xref,yref,ax,fig):
    
    #Imports data from the .dat file located at the specified path
    Tfile = open(path,'r')
    data = np.fromfile(Tfile,np.float32,-1) 
    c = data.reshape((-1,4))
    Tfile.close()
    
    #Defines labels for the axes corresponding to the four different sets of data available
    axislabels = ["DM", "Time", "s/n", "Width"]
    columns = np.hsplit(c,4) #x- and yref values dm=0, time=1, s/n=2, width=3
    
    #Creates the figure to be displayed. xref and yref corresponding to the chosen x and y values to be displayed
    ax.set_xlabel(axislabels[xref])
    ax.set_ylabel(axislabels[yref])
    ax.scatter(columns[xref], columns[yref], s = 7)
    ax.set_xlim(left = 0) #Sets lower x-limit to zero
    
    return fig
           
    
#Returns a matplotlib plot to be displayed on the canvas         
def plotimg(path,xref,yref):
    
    #Imports data from the .dat file located at the specified path
    file = open(path,'r')
    data = np.fromfile(file,np.float32,-1) 
    c = data.reshape((-1,4))
    file.close()

    #Defines labels for the axes corresponding to the four different sets of data available
    axislabels = ["DM", "Time", "s/n", "Width"]
    columns = np.hsplit(c,4) #x- and yref values dm=0, time=1, s/n=2, width=3
    
    #Creates the figure to be displayed. xref and yref corresponding to the chosen x and y values to be displayed
    fig = Figure(figsize = (8,6))
    ax = fig.add_subplot(111)
    ax.set_xlabel(axislabels[xref])
    ax.set_ylabel(axislabels[yref])
    ax.scatter(columns[xref], columns[yref], color = "0.7", alpha = 1, vmin = -1, s = 10, label = "RFI/Background")
    ax.set_xlim(left = 0) #Sets lower x-limit to zero
    ax.set_title(path.split('\\')[-1])
    return fig


#Returns a matplotlib plot to be displayed on the canvas         
def candPlot(path,xref,yref):
    excellent = []
    good = []
    least_acc = []
    
    figs = []
    
    #Imports data from the .dat file located at the specified path
    file = open(path,'r')
    data = np.fromfile(file,np.float32,-1) 
    c = data.reshape((-1,4))
    file.close()

    candFile = path.replace('.dat','_c.csv')
    #file = open(candFile,'r')
    dataset = pd.read_csv(candFile)

    X = dataset.iloc[:,1:6].values
    X = np.array(X)

    for i in range(len(X[:,4])):
        if X[:,4][i] == 3:
            excellent.append(X[i])
        elif X[:,4][i] == 2:
            good.append(X[i])
        elif X[:,4][i] == 1:
            least_acc.append(X[i])
    excellent = np.array(excellent)
    good = np.array(good)
    least_acc = np.array(least_acc)

    #Defines labels for the axes corresponding to the four different sets of data available
    axislabels = ["DM", "Time", "s/n", "Width"]
    columns = np.hsplit(c,4) #x- and yref values dm=0, time=1, s/n=2, width=3
    
    #Creates the figure to be displayed. xref and yref corresponding to the chosen x and y values to be displayed
    fig = Figure(figsize = (8,6))
    ax = fig.add_subplot(111)
    ax.set_xlabel(axislabels[xref])
    ax.set_ylabel(axislabels[yref])
    ax.scatter(columns[xref], columns[yref], color = "0.7", alpha = 1, vmin = -1, s = 6, label = "RFI/Background")
    ax.set_xlim(left = 0) #Sets lower x-limit to zero
    ax.set_title(path.split('\\')[-1])
    ax.legend()
    figs.append(fig)

    ax.set_xlim(left = 0) #Sets lower x-limit to zero
    ax.set_title(path)
    ax.legend()
    figs.append(fig)
    return figs[0], figs[1]