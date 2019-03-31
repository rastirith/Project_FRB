import numpy as np
import pandas as pd
import matplotlib as mpl
import glob, os, sys
import warnings
import math
from scipy import stats
from timeit import default_timer as timer

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

counter = 0     # Counter that counts the candidate number reaching the feature loop

pos_array_mp3 = [
        8, 20, 22, 24, 1, 3, 7, 30, 31, 74, 71, 101, 102,
        103, 104, 105, 107, 112, 114, 120, 121, 122, 124,
        125, 127, 128, 131, 137, 142, 145, 146, 147, 149,
        150, 153, 133, 151
        ]

pos_array = pos_array_mp3

source_paths = []   # Array of file paths to be reviewed

# Loops through all .dat files to store them in the 'source_paths' array
for file in glob.glob(os.getcwd() + '\idir\\train_1\\' + "*.dat"):
    source_paths.append(file)

shape_vals = []     # Array containing the shape feature values of the candidates
skew_vals = []      # Array containing the skewness feature values of the candidates
kurt_vals = []      # Array containing the kurtosis feature values of the candidates
kstest_vals = []    # Array containing the ks-test feature values of the candidates
reg_stats = []      # Array containing the reg slope feature values of the candidates
class_vals = []     # Array containing the classification labels of the candidates


timer1 = []
timer2 = []
timer3 = []
timer4 = []
timer5 = []
timer6 = []

y = 20

# Loops through the whole file space defined by 'source_paths'
for i in range(y): 
    #print(i)
    start5 = timer()
    progressBar(i,y)
    end5 = timer()
    timer5.append(end5 - start5)
    
    start1 = timer()
    
    path_index = i      # Current path index in the source_paths array
    file = source_paths[path_index]     # Setting which file to open
    
    start4 = timer()
    clusterData = cluster(file)
    
    newArr = clusterData[0]
    labels = clusterData[1]
    end4 = timer()
    timer4.append(end4 - start4)
    
    start6 = timer()
    for q in range(1,len(labels)):
        start2 = timer()
        
        label = labels[q]
        
        start3 = timer()
        features = writer(label, newArr, 'calc')
        end3 = timer()
        timer3.append(end3 - start3)
        
        # Adds the feature values to the corresponding arrays
        shape_vals.append(features[0])
        skew_vals.append(features[1])
        kurt_vals.append(features[2])
        kstest_vals.append(features[3])
        reg_stats.append(features[4])
        class_vals.append(features[-1])
        end2 = timer()
        timer2.append(end2 - start2)
    end6 = timer()
    timer6.append(end6 - start6)
    
    end1 = timer()
    timer1.append(end1 - start1)

progressBar(1,1)
print("\n")
print("Per file: ", np.mean(timer1))
print("Per candidate: ", np.mean(timer2))
print("Per features: ", np.mean(timer3))
print("Per clustering: ", np.mean(timer4))
print("Prog. bar: ", np.mean(timer5))
print("Per cluster loop: ", np.mean(timer6))

"""
# Creates dataframe table containing all feature values as well as the classification labels for each cluster
dataframe = pd.DataFrame({'Shape Feature': shape_vals,
                          'Skewness': skew_vals,
                          'Kurtosis': kurt_vals,
                          'KS-test stat': kstest_vals,
                          'Slope stat': reg_stats,  
                          'Label': class_vals})

dataframe.to_csv("feature_table.csv")   # Writes dataframe to .csv file"""


         
    
