import numpy as np
import matplotlib as mpl
import glob, os, sys
import warnings

from featuring import writer
from clustering import cluster

from timeit import default_timer as timer

warnings.filterwarnings("ignore", category=mpl.cbook.mplDeprecation)
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.set_printoptions(suppress=True)

def progressBar(value, endvalue, bar_length=20):
    
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    
    sys.stdout.write("\rPercent: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))  

source_paths = []   # Array of file paths to be reviewed

# Loops through all .dat files to store them in the 'source_paths' array
for file in glob.glob(os.getcwd() + '\\odir\\algTrain2\\' + "*.dat"):
    source_paths.append(file)
    
timer1 = []
timer2 = []
timer3 = []
timer4 = []

# Loops through the whole file space defined by 'source_paths'
for i in range(len(source_paths)): 
    
    start1 = timer()
    progressBar(i,len(source_paths))
    path_index = i      # Current path index in the source_paths array
    file = source_paths[path_index]     # Setting which file to open
    
    clusterData = cluster(file)
    
    newArr = clusterData[0]
    labels = clusterData[1]
    end1 = timer()
    timer1.append(end1 - start1)
    
    start2 = timer()
    timer4.append(len(labels))
    for q in range(1,len(labels)):
        start3 = timer()
        label = labels[q]
        result = writer(label, newArr, 'predict')
        end3 = timer()
        timer3.append(end3 - start3)
    end2 = timer()
    timer2.append(end2 - start2)

progressBar(1,1)


print("\n")
print("Clustering: ", np.mean(timer1))
print("Loop batch: ", np.mean(timer2))
print("Single loop: ", np.mean(timer3))
print("Len labels: ", np.mean(timer4))



         