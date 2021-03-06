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
import itertools

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

path = 42

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


#Noise condition for Nevents<Nmin =>noise
N_min = 20 #tuneable number (set to match Karako)
for q in range(1,len(np.unique(clusters))):
    if (len(labels_arr[q]) < N_min):
        for i in range(len(clusters)):
            if (clusters[i] == q - 1):
                clusters[i] = -1
#Re-order        
labels_arr = clusterSort(clusters, points)
clusterOrder(clusters)

      
#dm_lim = 40         # Increased DM-limit to check for clusters with 'fraction' of their points below this limit
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

ratios = []

# Condition for peak location
for q in range(1,len(np.unique(clusters))):

    signalToDm = list(zip(labels_arr[q][:,0], labels_arr[q][:,2]))
    signalToDm = np.array(signalToDm)
    min_val = min(signalToDm[:,1])
    #sharpness
    scaled_signal = preprocessing.MinMaxScaler().fit_transform(signalToDm)
    
    #y=0 for visualisation
    for i in range(len(signalToDm[:,1])):
        signalToDm[:,1][i] = signalToDm[:,1][i] - min_val
        
    #splitting into chunks
    split_param = 7 #parameter to determine splitting of cluster for analysis
    dummy = np.array_split(signalToDm,split_param)
    dummy2 = np.array_split(scaled_signal,split_param) #sharpness
    
    meanSN = []
    meanDM = []

    s_meanSN = []
    s_meanDM = []
    for i in range(len(dummy)):
        tempSN = np.mean(dummy[i][:,1])
        tempDM = np.mean(dummy[i][:,0])
        meanSN.append(tempSN)
        meanDM.append(tempDM)
        #testing sharpness
        s_tempSN = np.mean(dummy2[i][:,1])
        s_tempDM = np.mean(dummy2[i][:,0])
        s_meanSN.append(s_tempSN)
        s_meanDM.append(s_tempDM)
    
    max_val = max(meanSN + min_val)
    
    
    #Condition for location of peak S/N
    max_ind = np.argmax(meanSN)
    if (max_ind > 4) or (max_ind < 2):
        for i in range(len(clusters)):
            if (clusters[i] == q - 1):
                clusters[i] = -1
    
       #developing peak shape conditions
    else:
        weight_1 = 1/3
        weight_2 = 2/3
        weight_3 = -0.1
        check_1 = 0.075
        check_2 = 0.15
        score = [1,2,1.5,1.5]
        rating = 0
    
    sub_arr1 = meanDM[0:max_ind + 1]
    sub_arr2 = meanDM[max_ind:len(meanDM)]
    
    diffs1 = [abs(e[1] - e[0]) for e in itertools.permutations(sub_arr1, 2)]
    diffs2 = [abs(e[1] - e[0]) for e in itertools.permutations(sub_arr2, 2)]
    
    avg_diff1 = sum(diffs1)/len(diffs1)
    avg_diff2 = sum(diffs2)/len(diffs2)
    
    diffDiff = abs(avg_diff1 - avg_diff2)
    sumDiff = avg_diff1 + avg_diff2
    diffRatio = diffDiff/sumDiff
    
    for i in range(max_ind - 1, -1, -1):
        ratio=meanSN[i]/meanSN[i+1]
    
        if ((ratio>=(1-check_1)) and (ratio<=1)):
            rating += weight_1*score[max_ind-(i+1)]
        elif ((ratio>=(1-check_2)) and (ratio<=1)):
            rating += weight_2*score[max_ind-(i+1)]
        elif ratio <=1:
            rating += score[max_ind-(i+1)]
        else:
            rating += weight_3*score[max_ind-(i+1)]
    
    for i in range((max_ind+1),split_param):
        ratio=meanSN[i]/meanSN[i-1]
    
        if ((ratio>=(1-check_1)) and (ratio<=1)):
            rating += weight_1*score[i-max_ind-1]
        elif ((ratio>=(1-check_2)) and (ratio<=1)):
            rating += weight_2*score[i-max_ind-1]
        elif ratio <=1:
            rating += score[i-max_ind-1]
        else:
               rating += weight_3*score[i-max_ind-1]
    confidence = rating/9
    SN_conf = 0.9828639 + ((-0.9828632502)/(1+((max_val/4.630117)**1.341797)))
    diff_conf = 0.2 + ((0.46)/(1+((diffRatio/0.5949996)**1450.932)))
    tot_conf = 0.25*SN_conf + 0.15* diff_conf + 0.60*confidence
    
    #sharpness
    diff_SN = max(s_meanSN) - min(s_meanSN)
    diff_DM = s_meanDM[-1] - s_meanDM[0] #?????center this around peak
    sharp_ratio = diff_SN/diff_DM #height/width
    ratios.append(sharp_ratio)
        
    diffs1 = [abs(e[1] - e[0]) for e in itertools.permutations(sub_arr1, 2)]
    diffs2 = [abs(e[1] - e[0]) for e in itertools.permutations(sub_arr2, 2)]
        
    avg_diff1 = sum(diffs1)/len(diffs1)
    avg_diff2 = sum(diffs2)/len(diffs2)
        
    print(round(avg_diff1,1))
    print(round(avg_diff2,1))
        
        
    xwidth=(min(signalToDm[:,0]) - max(signalToDm[:,0]))/12
        
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.bar(meanDM,meanSN, align='center',width=xwidth, alpha=0.2)
    #plt.show()
    ax1.set_title(str(max_val) + "\n" + str(confidence) + "\n" + str(SN_conf) + "\n" + str(tot_conf))
    ax = fig.add_subplot(111)
    ax.scatter(signalToDm[:,0], signalToDm[:, 1], alpha = 0.4, vmin = -1, s = 10)
    #ax.plot(x,y)
    ax.set_xlabel("DM")
    ax.set_ylabel("S/N")
    plt.show()    
        
        
#Re-order        
labels_arr = clusterSort(clusters, points)
clusterOrder(clusters)                



#y = 0.9812087 + ((-0.981208633)/((x/6.881581)**1.620657)) # S/N conf function
            
"""                
#Plotting for visualisation of S/N             
for q in range(1,len(np.unique(clusters))):    
    signalToDm = list(zip(labels_arr[q][:,0], labels_arr[q][:,2]))
    signalToDm = np.array(signalToDm)
    min_val = min(signalToDm[:,1])
    #y=0 for visualisation
    for i in range(len(signalToDm[:,1])):
        signalToDm[:,1][i] = signalToDm[:,1][i] - min_val
        
    #splitting into chunks
    dummy = np.array_split(signalToDm,split_param)
    
    meanSN = []
    meanDM = []
    for i in range(len(dummy)):
        tempSN = np.mean(dummy[i][:,1])
        tempDM = np.mean(dummy[i][:,0])
        meanSN.append(tempSN)
        meanDM.append(tempDM)
     
    xwidth=(min(signalToDm[:,0]) - max(signalToDm[:,0]))/12
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.bar(meanDM,meanSN, align='center',width=xwidth, alpha=0.2)
    #plt.show()
    ax1.set_title(np.argmax(meanSN))
    ax = fig.add_subplot(111)
    ax.scatter(signalToDm[:,0], signalToDm[:, 1], alpha = 0.4, vmin = -1, s = 10)
    #ax.plot(x,y)
    ax.set_xlabel("DM")
    ax.set_ylabel("S/N")
    plt.show()
"""           
#Plotting DM TIME w/clusters
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(points[:, 1], points[:, 0], c=clusters, cmap="Paired", alpha = 0.4, vmin = -1, s = 10)



"""
    signal_scaled = preprocessing.StandardScaler().fit_transform(signalToDm)
    #print(signal_scaled)
    
    #y=0
    for i in range(len(signal_scaled[:,1])):
        signal_scaled[:,1][i] = signal_scaled[:,1][i] - min(signal_scaled[:,1])
     
    max_val = max(signal_scaled[:,1])
    print(signal_scaled)
    mu = 0
    variance = 1
    sigma = math.sqrt(variance)
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 1001)
    y = mlab.normpdf(x, mu, sigma)
    for i in range(len(y)):
        y[i] = y[i]*2*max_val

        
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
