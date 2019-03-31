import numpy as np
import matplotlib as mpl
import glob, os, sys
import warnings

from featuring import writer
from clustering import cluster

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

# Loops through the whole file space defined by 'source_paths'
for i in range(len(source_paths)): 
    progressBar(i,len(source_paths))
    path_index = i      # Current path index in the source_paths array
    file = source_paths[path_index]     # Setting which file to open
    
    clusterData = cluster(file)
    
    newArr = clusterData[0]
    labels = clusterData[1]
    
    for q in range(1,len(labels)):
        label = labels[q]
        result = writer(label, newArr, 'predict')

progressBar(1,1)
         