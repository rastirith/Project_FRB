import numpy as np
from matplotlib.figure import Figure

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