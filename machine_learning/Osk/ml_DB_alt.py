import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.mlab as mlab
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
import glob, os
from sklearn import preprocessing
import scipy.cluster.hierarchy as hcluster 
import sys
import math
import warnings
warnings.filterwarnings("ignore",category=mpl.cbook.mplDeprecation)

def progressBar(value, endvalue, bar_length=20):
    
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    
    sys.stdout.write("\rPercent: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))

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

#creates dataframe for file
def DF(path):
    axislabels = ["DM", "Time", "S/N", "Width"]
    Tfile = open(path,'r')
    data = np.fromfile(Tfile,np.float32,-1) 
    c = data.reshape((-1,4))
    df = pd.DataFrame(c,columns=axislabels)
    Tfile.close()
    return df

def mean(arr):
    xsum = 0
    for i in (arr):
        xsum += i
    mean = xsum/len(arr)
    return mean

def variance(arr,avg):
    xsum = 0
    for i in (arr):
        xsum += (i - avg)**2
    var = xsum/len(arr)
    return var

#array of file locations and chosing the file to inspect with path
source_paths = []
path = 48
#filling source_paths from the idir
for file in glob.glob(os.getcwd() + '\idir\\' + "*.dat"):
    source_paths.append(file)
    
    
#setting which file to open
FILE = source_paths[path]
#getting df for test file
df = DF(FILE)
#getting arrays from df
orig_X = np.array(df)
X_db = np.array(df.drop(columns=['Width', 'S/N']))
X = np.array(df.drop(columns=['Width']))

#sorted by DM
points_db = X_db[X_db[:,0].argsort()]
points = X[X[:,0].argsort()]

dm_lim = 0.03*max(points_db[:,0])
points_new = []
print("Dm limit 1: " + str(round(dm_lim,1)))

for i in range(len(points_db)):
    if (points_db[i][0] > dm_lim):
        points_new.append(points_db[i])
       
points_new = np.array(points_new)

X_scaled = preprocessing.MinMaxScaler().fit_transform(points_new) #Rescales the data so that the x- and y-axes get ratio 1:1

xeps = 0.025    # Radius of circle to look around for additional core points
xmin = 2        # Number of points within xeps for the point to count as core point

clusters = DBSCAN(eps=xeps, min_samples = xmin).fit_predict(X_scaled)   
#plt.scatter(X_scaled[:, 1], X_scaled[:, 0], c=clusters, cmap="Paired", alpha = 0.4, vmin = -1, s = 15)

# Re-inserts bottom points with labels -1 for RFI
length = len(points) - len(clusters)
clusters = np.insert(clusters,0,np.full(length,-1))

labels_arr = clusterSort(clusters, points)
clusterOrder(clusters)

dm_lim = 40         # Increased DM-limit to check for clusters with 'fraction' of their points below this limit
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

for q in range(1,len(np.unique(clusters))):
    
   signalToDm = list(zip(labels_arr[q][:,0], labels_arr[q][:,2]))
   signalToDm = np.array(signalToDm)
   signal_scaled = preprocessing.StandardScaler().fit_transform(signalToDm)
   #print(signal_scaled)
    
    
   #splitting into chunks
   #print(len(signalToDm))

   dummy = np.array_split(signalToDm,5)
   #print(len(dummy))
   #for i in range(8):
   #    print(dummy[i][:,1])
   avg = []
   for i in range(len(dummy)):
       temp = np.mean(dummy[i][:,1])
       avg.append(temp)
   print(avg)
    
    
   for i in range(len(signal_scaled[:,1])):
       signal_scaled[:,1][i] = signal_scaled[:,1][i] - min(signal_scaled[:,1])
        
   max_val = max(signal_scaled[:,1])
   #ratio = max_val/
    
    
   #average = mean(labels_arr[q][:,0])
   #var = variance(labels_arr[q][:,0], average)

   #temp_dm = labels_arr[q][:,0]
   #temp_s = labels_arr[q][:,2]

   sigma = math.sqrt(1)
   x = np.linspace(0 - 3*sigma, 0 + 3*sigma, 1001)
   y = mlab.normpdf(x, 0, sigma)
    
    #ratio = max(labels_arr[q][:,2])/max(y)
    
   for i in range(len(y)):
       y[i] = y[i]*max_val*2
        
   xsum = 0
   """
   for i in range(len(labels_arr[q][:,0])):
        #temp_dm = signal_scaled[:,0][i] - min(signal_scaled[:,0])
        #frac = temp_dm/(max(labels_arr[q][:,0]) - min(signal_scaled[:,0]))
        #y_temp = y[int(round(frac*1000))]
        print(signal_scaled[:,0])
        print("mod: " + str(y_temp))        
        #term = ((signal_scaled[:,1][i] - y_temp)**2)/y_temp
        #xsum += term"""
    
    #red_chi = xsum/len(signal_scaled[:,1])
    #print(red_chi)
        
    
   fig = plt.figure()
   ax = fig.add_subplot(111)
   ax.scatter(signal_scaled[:,0], signal_scaled[:,1], alpha = 0.4, vmin = -1, s = 10)
   ax.plot(x,y)
   ax.set_xlabel("DM")
   ax.set_ylabel("S/N")
   #plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(points[:, 1], points[:, 0], c=clusters, cmap="Paired", alpha = 0.4, vmin = -1, s = 10)





"""
time_diff = 0.05
for q in range(1,len(np.unique(clusters))):
    quantile_diff = np.quantile(labels_arr[q][:,1], 0.75) - np.quantile(labels_arr[q][:,1], 0.25)
    if (quantile_diff > time_diff):
        for i in range(len(clusters)):
            if (clusters[i] == q - 1):
                clusters[i] = -1"""

#fig = plt.figure()
#ax = fig.add_subplot(111,projection = '3d')
#ax = fig.add_subplot(111)
#ax.scatter(points[:, 1], points[:, 0], zs, c=clusters, cmap="Paired", alpha = 0.4, vmin = -1, s = 10)
#ax.scatter(points[:, 1], points[:, 0], c=clusters, cmap="Paired", alpha = 0.4, vmin = -1, s = 10)

print("Dm limit 2: " + str(dm_lim))
print("Old array length: " + str(len(points)))
print("New array length: " + str(len(points_new)))

#plt.xlabel("Time")
#plt.ylabel("DM")
plt.title(source_paths[path])

plt.show()
