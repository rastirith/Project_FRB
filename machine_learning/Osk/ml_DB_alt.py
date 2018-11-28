import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
import glob, os
from sklearn import preprocessing
import scipy.cluster.hierarchy as hcluster 
import sys


def progressBar(value, endvalue, bar_length=20):
    
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    
    sys.stdout.write("\rPercent: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))

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

def clusterOrder(clusterArr):
    
    lab_arr = np.unique(clusterArr)
    
    for i in range(len(clusterArr)): 
        for q in range(1,len(lab_arr)):
            if (lab_arr[q] == clusterArr[i]):
                clusterArr[i] = q - 1
                break

def clusterSort(clusterArr, pointsArr):
    temp_arr = []     # Array of arrays containing points for each label. e.g. labels_arr[0] contains all points with label -1
    lab_arr = np.unique(clusterArr)

    for i in range(len(np.unique(clusterArr))):
        temp_list = []
        temp_arr.append(temp_list)
    
    for i in range(len(clusterArr)):
        for q in range(len(lab_arr)):
            if (clusterArr[i] == lab_arr[q]):
                temp_arr[q].append(pointsArr[i])
                break

    for q in range(len(temp_arr)):
        temp_arr[q] = np.array(temp_arr[q])
    
    temp_arr = np.array(temp_arr)
    return temp_arr

source_paths = []
path = 48

for file in glob.glob(os.getcwd() + '\idir\\' + "*.dat"):
    source_paths.append(file)
    
x = val(source_paths[path],0) #DM
y = val(source_paths[path],1) #Time

points = list(zip(x, y))

points = np.array(points)
points = points[points[:,0].argsort()]

dm_lim = 0.03*max(x)
points_new = []
print("Dm limit 1: " + str(round(dm_lim,1)))

for i in range(len(points)):
    if (points[i][0] > dm_lim):
        points_new.append(points[i])
       
points_new = np.array(points_new)

X_scaled = preprocessing.MinMaxScaler().fit_transform(points_new) #Rescales the data so that the x- and y-axes get ratio 1:1

xeps = 0.025    # Radius of circle to look around for additional core points
xmin = 5        # Number of points within xeps for the point to count as core point

clusters = DBSCAN(eps=xeps, min_samples = xmin).fit_predict(X_scaled)   
#plt.scatter(X_scaled[:, 1], X_scaled[:, 0], c=clusters, cmap="Paired", alpha = 0.4, vmin = -1, s = 15)

# Re-inserts bottom points with labels -1 for RFI
length = len(points) - len(clusters)
clusters = np.insert(clusters,0,np.full(length,-1))

labels_arr = clusterSort(clusters, points)
clusterOrder(clusters)

dm_lim = 70         # Increased DM-limit to check for clusters with 'fraction' of their points below this limit
fraction = 0.05 

# Condition that sets all clusters with 'fraction' of its points below dm_lim to also be classified as RFI
for q in range(1,len(np.unique(clusters))):
    num_temp = int(round(fraction*len(labels_arr[q]),0))    # Calculates how many points of a label a certain fraction corresponds to and rounds to nearest integer
    temp = sort(labels_arr[q][:,0],num_temp)                # Returns the 'num_temp' lowest dms in labels_arr[q]

    if ((len(temp) > 0) and (max(temp) < dm_lim)):          # If the highest number in temp is below dm_lim then so is the rest in 'temp'

        for i in range(len(clusters)):
            if (clusters[i] == q - 1):
                clusters[i] = -1
                
# Condition that sets all points with dm below dm_lim to be classified as RFI    
for i in range(len(clusters)):
    if (points[i][0] <= dm_lim):
        clusters[i] = -1
 

labels_arr = clusterSort(clusters, points)
clusterOrder(clusters)

time_diff = 0.05
for q in range(1,len(np.unique(clusters))):
    quantile_diff = np.quantile(labels_arr[q][:,1], 0.75) - np.quantile(labels_arr[q][:,1], 0.25)
    if (quantile_diff > time_diff):
        for i in range(len(clusters)):
            if (clusters[i] == q - 1):
                clusters[i] = -1


plt.scatter(points[:, 1], points[:, 0], c=clusters, cmap="Paired", alpha = 0.4, vmin = -1, s = 10)

print("Dm limit 2: " + str(dm_lim))
print("Old array length: " + str(len(points)))
print("New array length: " + str(len(points_new)))

plt.xlabel("Time")
plt.ylabel("DM")
plt.title(source_paths[path])

plt.show()
