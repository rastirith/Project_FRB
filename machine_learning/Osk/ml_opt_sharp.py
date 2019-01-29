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
import warnings
import itertools
import random

warnings.filterwarnings("ignore",category=mpl.cbook.mplDeprecation)
warnings.filterwarnings("ignore",category=RuntimeWarning)

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

#creates dataframe for file
def DF(path):
    axislabels = ["DM", "Time", "S/N", "Width"]
    Tfile = open(path,'r')
    data = np.fromfile(Tfile,np.float32,-1) 
    c = data.reshape((-1,4))
    df = pd.DataFrame(c,columns=axislabels)
    Tfile.close()
    return df

step = 0.01

sharp_step = 0.1
sharp_range = 10
sharpsteps_1d = int(sharp_range/sharp_step)
sharp_iterations = int((sharpsteps_1d + 1)**2)




#iter_arr = np.zeros(200)
sharp_struct = []
TP_iter = np.zeros((int(1/step), sharp_iterations))
FP_iter = np.zeros((int(1/step), sharp_iterations))
TN_iter = np.zeros((int(1/step), sharp_iterations))
FN_iter = np.zeros((int(1/step), sharp_iterations))

for i in range(int(sharpsteps_1d + 1)):
    a = i*sharp_step
    print(a)
    for k in range(int(sharpsteps_1d + 1)):
        b = k*sharp_step
        sharp_struct.append([a,b])

sharp_struct = np.array(sharp_struct)


counter = 0
true_pos = np.zeros(int(1/step))
false_pos = np.zeros(int(1/step))
true_neg = np.zeros(int(1/step))
false_neg = np.zeros(int(1/step))

pos_array = [
        1, 3, 7, 8, 10, 14, 19, 20, 22, 24, 30, 31, 38,
        39, 61, 71, 74, 90, 96, 97, 99, 100, 101, 102,
        103, 104, 105, 107, 111, 112, 114, 120, 121, 122,
        124, 125, 127, 128, 131, 133, 137, 142, 145, 146,
        147, 149, 150, 151, 152, 153
        ]

#array of file locations and chosing the file to inspect with path
source_paths = []

#filling source_paths from the idir
for file in glob.glob(os.getcwd() + '\idir\\' + "*.dat"):
    source_paths.append(file)
    
ratios = []

for i in range(0,72): 
    print("\n" + str(i))
    para = 0
    path=i
    #setting which file to open
    FILE = source_paths[path]
    #getting df for test file
    df = DF(FILE)
    #getting arrays from df
    orig_X = np.array(df)
    #print(len(orig_X))
    X_db = np.array(df.drop(columns=['Width', 'S/N']))
    X = np.array(df.drop(columns=['Width']))
    
    #sorted by DM
    points_db = X_db[X_db[:,0].argsort()]
    points = X[X[:,0].argsort()]
    
    dm_lim = 0.03*max(points_db[:,0])
    points_new = []
    #print("Dm limit 1: " + str(round(dm_lim,1)))
    
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
    
    # Condition for peak location
    # Condition for peak location
    clustcount = len(np.unique(clusters))
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
            counter += 1
            
            for h in range(len(sharp_struct)):
                #progressBar(h,len(score_struct))
                
                SnWeight = sharp_struct[h][0]
                DmWeight = sharp_struct[h][1]
                rating = 0
                
                sub_arr1 = meanDM[0:max_ind + 1]
                sub_arr2 = meanDM[max_ind:len(meanDM)]
                  
                #sharpness
                diff_SN = max(s_meanSN) - (0.5*s_meanSN[0] + 0.5*s_meanSN[-1])
                diff_DM = s_meanDM[-1] - s_meanDM[0] #?????center this around peak
                sharp_ratio = (SnWeight*diff_SN)/((DmWeight*diff_DM) + (SnWeight*diff_SN))#height/width
                ratios.append(sharp_ratio)    
                tot_conf = sharp_ratio
                #int(0.5*len(true_pos)),int(0.8*len(true_pos))
                for i in range(len(true_pos)):
                
                    if ((tot_conf >= i*step) and (counter in pos_array)):
                        true_pos[i] += 1
                        TP_iter[i][h] += 1
                    elif ((tot_conf >= i*step) and (counter not in pos_array)):
                        false_pos[i] += 1
                        FP_iter[i][h] += 1
                    elif ((tot_conf < i*step) and (counter not in pos_array)):
                        true_neg[i] += 1
                        TN_iter[i][h] += 1
                    elif ((tot_conf < i*step) and (counter in pos_array)):
                        false_neg[i] += 1
                        FN_iter[i][h] += 1
                        
            progressBar(q, clustcount)
            
    #Re-order        
    labels_arr = clusterSort(clusters, points)
    clusterOrder(clusters)                  

x_val = np.arange(0,1,step)

T_pos=[]
F_neg=[] 
T_neg=[] 
F_pos=[]

xTP = np.zeros((int(1/step),sharp_iterations))
xFN = np.zeros((int(1/step),sharp_iterations))
xTN = np.zeros((int(1/step),sharp_iterations))
xFP = np.zeros((int(1/step),sharp_iterations))

for i in range(len(TP_iter)):
    for h in range(sharp_iterations):
        
        if (TP_iter[i][h] + FN_iter[i][h]) != 0:
            xTP[i][h] = 100*TP_iter[i][h]/(TP_iter[i][h] + FN_iter[i][h])
            xFN[i][h] = 100*FN_iter[i][h]/(TP_iter[i][h] + FN_iter[i][h])
        if (TN_iter[i][h] + FP_iter[i][h]) != 0:
            xTN[i][h] = 100*TN_iter[i][h]/(TN_iter[i][h] + FP_iter[i][h])
            xFP[i][h] = 100*FP_iter[i][h]/(TN_iter[i][h] + FP_iter[i][h])       

sharp_setting = 0  
max_avg = 0   
for h in range(sharp_iterations):
    dist_avg = 0
    for i in range(len(TP_iter)):
        dist_avg += (xTP[i][h] - xFP[i][h])/(len(TP_iter))
    if (dist_avg  > max_avg):
        max_avg = dist_avg
        sharp_setting = h
        #print(h)
        #print(dist_avg)
        #print(sharp_struct[h])
"""   
dist = 0   
for i in range (len(TP_iter)):       
    if ((xTP[i][sharp_setting] - xFP[i][sharp_setting]) > dist):
        dist = (xTP[i][h] - xFP[i][h])
        conf_setting = i"""

print("Score setting: " + str(sharp_struct[sharp_setting]))



"""    
for i in range(len(TP_iter)):
    for h in range(score_iterations):
        xTP.append(round(100*TP_iter[i][h]/(TP_iter[i][h] + FN_iter[i][h]), 1))
        xFN.append(round(100*FN_iter[i][h]/(TP_iter[i][h] + FN_iter[i][h]), 1))
        xTN.append(round(100*TN_iter[i][h]/(TN_iter[i][h] + FP_iter[i][h]), 1))
        xFP.append(round(100*FP_iter[i]/(TN_iter[i][h] + FP_iter[i][h]), 1))"""




"""
for i in range(int(1/step)):   
    
    #rates?
    T_pos.append(round(100*true_pos[i]/(true_pos[i] + false_neg[i]), 1))
    F_neg.append(round(100*false_neg[i]/(true_pos[i] + false_neg[i]), 1))
    T_neg.append(round(100*true_neg[i]/(true_neg[i] + false_pos[i]), 1))
    F_pos.append(round(100*false_pos[i]/(true_neg[i] + false_pos[i]), 1))"""


