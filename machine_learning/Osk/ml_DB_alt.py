import numpy as np
import pandas as pd
import sys
import itertools
from scipy import stats
from matplotlib import pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import DBSCAN
from matplotlib.figure import Figure
from sklearn.mixture import GaussianMixture as GMM
from math import ceil
import glob, os
from scipy import linalg
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as hcluster 




def progressBar(value, endvalue, bar_length=20):

        percent = float(value) / endvalue
        arrow = '-' * int(round(percent * bar_length)-1) + '>'
        spaces = ' ' * (bar_length - len(arrow))

        sys.stdout.write("\rPercent: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
        sys.stdout.flush()


def ind_avg(num, ind_arr, val_arr):
    sum = 0
    for i in range(num):
        sum += val_arr[ind_arr[i]]
        
    avg = sum/num
    return avg
        
def norm_avg(num,arr):
    sum = 0
    if (num > 0):
        for i in range(num):
            sum += arr[i]
    elif (num < 0):
        num = -num
        for i in range(1, num + 1):
            sum += arr[-i]
    
    avg = sum/num
    return avg

def val(path, ref):
    
    #Imports data from the .dat file located at the specified path
    Tfile = open(path,'r')
    data = np.fromfile(Tfile,np.float32,-1) 
    c = data.reshape((-1,4))
    Tfile.close()
    refarr = []
    #Defines labels for the axes corresponding to the four different sets of data available
    #axislabels = ["DM", "Time", "StoN", "Width"]
    columns = np.hsplit(c,4) #x- and yref values dm=0, time=1, ston=2, width=3
    for i in range(len(columns[ref])):
        refarr.append(columns[ref][i][0])
        
    return refarr

# Returns 'num' lowest elements of array 'arr' in a new array
def sort(arr,num):
    xsorted = np.sort(arr)[:num]
    return xsorted



source_paths = []
path = 42

for file in glob.glob(os.getcwd() + '\idir\\' + "*.dat"):
    source_paths.append(file)
    
x = val(source_paths[path],0)
y = val(source_paths[path],1)

points = list(zip(x, y))

points = np.array(points)
points = points[points[:,0].argsort()]
print(len(points))
up = ceil(points[-1][1])

dm_lim = 0.05*max(x)

points_new = []

for i in range(len(points)):
    if (points[i][0] > dm_lim):
        points_new.append(points[i])
       
points_new = np.array(points_new)
X_scaled = StandardScaler().fit_transform(points_new)

xeps = 0.05*max(X_scaled[:,0])
xmin = 5

clusters = DBSCAN(eps=xeps, min_samples = xmin).fit_predict(X_scaled)

# Re-inserts bottom points with labels -1 for RFI
length = len(points) - len(clusters)
clusters = np.insert(clusters,0,np.full(length,-1))

labels_arr = []
index_arr = []

# Puts points of different labels in different arrays in 'labels_arr'
# Array at index 0 contains all points with label -1, array at index 1 contains all points with label 0 etc.
for p in range(len(np.unique(clusters))):
    temp_points = []
    temp_ind = []
    for i in range(len(clusters)):
        if (p - 1 == clusters[i]):
            temp_points.append(points[i])
            temp_ind.append(i)
    labels_arr.append(np.array(temp_points))
    #index_arr.append(np.array(temp_ind))

labels_arr = np.array(labels_arr)
#index_arr = np.array(index_arr)

# Condition that sets all points with dm below dm_lim to be classified as RFI
for i in range(len(clusters)):
    
    if (points[i][0] <= dm_lim):
        clusters[i] = -1


fraction = 0.1    
# Condition that sets all clusters with 'fraction' of its points below dm_lim to also be classified as RFI
for q in range(1,len(np.unique(clusters))):
    num_temp = int(round(fraction*len(labels_arr[q]),0))    # Calculates how many points of a label a certain fraction corresponds to and rounds to nearest integer
    temp = sort(labels_arr[q][:,0],num_temp)                # Returns the 'num_temp' lowest dms in labels_arr[q]
    if ((len(temp) < 0) and (max(temp) < dm_lim)):
        for i in range(len(clusters)):
            if (clusters[i] == q - 1):
                clusters[i] = -1




"""
for p in range(len(np.unique(clusters))):
    print("p: " + str(p))
    for i in range(len(labels_arr[p])):
        if (points[i][0] <= dm_lim):
            print(index_arr[p][i])
            clusters[index_arr[p][i]] = -1"""

#clusters = np.array(list(map(lambda x: x + 2, clusters)))
#print(np.unique(clusters))
plt.scatter(points[:, 1], points[:, 0], c=clusters, cmap="brg", alpha = 0.4, vmin = -1, s = 15) 

print("Dm limit: " + str(dm_lim))
print("Old array length: " + str(len(points)))
print("New array length: " + str(len(points_new)))

plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.title(source_paths[path])
for i in range(up + 1):
    plt.axvline(i,alpha = 0.2)
    
plt.show()
