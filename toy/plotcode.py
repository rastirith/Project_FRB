import matplotlib.pyplot as plt
import numpy as np
import glob, os



def plotimg(xref, yref, columns):
    
    fig = plt.figure()
    
    ax1 = fig.add_subplot(221)
    ax1.plot(columns[1], columns[0], 'ro', markersize = 2)
    ax1.set_xlabel(axislabels[1])
    ax1.set_ylabel(axislabels[0]) 
    xlim1 = ax1.get_xlim()
    ylim1 = ax1.get_ylim()
    
    ax2 = fig.add_subplot(222)
    ax2.plot(columns[1], columns[2], 'ro', markersize = 2)
    ax2.set_xlabel(axislabels[1])
    ax2.set_ylabel(axislabels[2])
    xlim2 = ax2.get_xlim()
    ylim2 = ax2.get_ylim()
    
    ax3 = fig.add_subplot(223)
    ax3.plot(columns[0], columns[2], 'ro', markersize = 2)
    ax3.set_xlabel(axislabels[0])
    ax3.set_ylabel(axislabels[2])
    xlim3 = ax3.get_xlim()
    ylim3 = ax3.get_ylim()
    
    ax4 = fig.add_subplot(224)
    ax4.plot(columns[1], columns[2], 'ro', markersize = 2)  
    ax4.set_xlabel(axislabels[1])
    ax4.set_ylabel(axislabels[2])
    xlim4 = ax4.get_xlim()
    ylim4 = ax4.get_ylim()
    
    def enter_axes(event):
        global ref
        print('enter_axes' + str(event.inaxes))
        event.canvas.figure.patch.set_color('blue')
        event.canvas.draw()

    fig.canvas.mpl_connect('axes_enter_event', enter_axes)
    #fig1.FigureCanvasBase.mpl_connect('axes_leave_event', leave_axes)
    
    fig.tight_layout()



idir_path = os.getcwd() + "\\idir"
   
source_paths = []       #List of filenames to be viewed in program
source_ind = 0
xref = 1
yref = 0


for file in glob.glob(idir_path + "/" + "*.dat"):
        source_paths.append(file)

#Imports data from the .dat file located at the specified path
Tfile = open(source_paths[source_ind],'r')
data = np.fromfile(Tfile,np.float32,-1) 
c = data.reshape((-1,4))
Tfile.close()

#Defines labels for the axes corresponding to the four different sets of data available
axislabels = ["DM", "Time", "s/n", "Width"]
columns = np.hsplit(c,4) #x- and yref values dm=0, time=1, s/n=2, width=3

plotimg(xref, yref, columns)