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
from scipy.stats import skew, kurtosis
import math
import pickle
import time

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


def ks_cordes(dmArr,snArr,timeArr,peakDmMean,freq = 0.334,bandWidth = 64):
    cordes = []
    peakSN = max(snArr)
    snFreqArr = []
    cordesFreqArr = []
    
    Wms = np.percentile(timeArr,75)-np.percentile(timeArr,25)
    Wms = Wms*1000
    dmScaled = dmArr - peakDmMean
    snRatios = snArr/peakSN
    
    x = np.linspace(min(dmScaled),max(dmScaled),2000)
    zeta = (6.91*10**-3)*bandWidth*(freq**-3)*(Wms**-1)*x
    
    for i in range(len(x)):
        cordes.append((math.pi**(1/2))*0.5*(zeta[i]**-1)*math.erf(zeta[i]))

    
    for i in range(len(snRatios)):
        temp_arr = []
        frequency = int(snRatios[i]*1000)              # Needs to be an integer, timed it by 1000 to reduce rounding errors, proportions still same
        temp_arr = [dmScaled[i]] * frequency   # Creates the corresponding number of elements and adds it to the array
        snFreqArr.extend(temp_arr)
    
    for i in range(len(cordes)):
        temp_arr = []
        frequency = int(cordes[i]*1000)
        temp_arr = [x[i]] * frequency
        cordesFreqArr.extend(temp_arr)

    statistic = stats.ks_2samp(snFreqArr,cordesFreqArr)
    return statistic[0]


counter = 0
true_pos = 0
false_pos = 0
true_neg = 0
false_neg = 0

pos_array_mp2 = [
        1, 3, 7, 8, 10, 14, 19, 20, 22, 24, 30, 31, 38,
        39, 61, 71, 74, 90, 96, 97, 99, 100, 101, 102,
        103, 104, 105, 107, 111, 112, 114, 120, 121, 122,
        124, 125, 127, 128, 131, 133, 137, 142, 145, 146,
        147, 149, 150, 151, 152, 153
        ]

pos_array_mp3 = [
        8, 20, 22, 24, 1, 3, 7, 30, 31, 74, 71, 101, 102,
        103, 104, 105, 107, 112, 114, 120, 121, 122, 124,
        125, 127, 128, 131, 137, 142, 145, 146, 147, 149,
        150, 153, 133, 151
        ]



pos_array = pos_array_mp3


#array of file locations and chosing the file to inspect with path
source_paths = []

#filling source_paths from the idir
for file in glob.glob(os.getcwd() + '\idir\\' + "*.dat"):
    source_paths.append(file)
    
ratios = []

shape_vals = []
sharp_vals = []
skew_vals = []
kurt_vals = []
kstest_vals = []
class_vals = []
cand_paths = []


for i in range(0,5): 
    print(i)
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
    
    xeps = 0.025     # Radius of circle to look around for additional core points
    xmin = 3         # Number of points within xeps for the point to count as core point
    
    clusters = DBSCAN(eps=xeps, min_samples = xmin, n_jobs = -1).fit_predict(X_scaled)  
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
    
    rejected = []
    least_acc = []
    good = []
    excellent = []
    rfi = []
    
    # Condition for peak location
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
        
        
        # Condition for location of peak S/N
        max_ind = np.argmax(meanSN)
        peakMeanDm = meanDM[max_ind]
        if (max_ind > 4) or (max_ind < 2):
            for i in range(len(clusters)):
                if (clusters[i] == q - 1):
                    clusters[i] = -1
        
        # Developing peak shape conditions
        else:
            counter += 1
            freq_arr = []
            
            weight_1 = -1
            weight_2 = -0.3
            weight_3 = 1
            weight_4 = -1
            check_1 = 0.075
            check_2 = 0.15
            score = [0,1.3,2.5,2.5]
            max_score = 2*(score[0] + score[1] + score[2])
            
            rating = 0
            
            for i in range(max_ind - 1, -1, -1):
                ratio=meanSN[i]/meanSN[i+1]
            
                if ((ratio>=(1-check_1)) and (ratio<=1)):
                    rating += weight_1*score[max_ind-(i+1)]
                elif ((ratio>=(1-check_2)) and (ratio<=1)):
                    rating += weight_2*score[max_ind-(i+1)]
                elif (ratio<=1):
                    rating += weight_3*score[max_ind-(i+1)]
                else:
                    rating += weight_4*score[max_ind-(i+1)]

            for i in range((max_ind+1),split_param):
                ratio=meanSN[i]/meanSN[i-1]

                if ((ratio>=(1-check_1)) and (ratio<=1)):
                    rating += weight_1*score[i-max_ind-1]
                elif ((ratio>=(1-check_2)) and (ratio<=1)):
                    rating += weight_2*score[i-max_ind-1]
                elif ratio <=1:
                    rating += weight_3*score[i-max_ind-1]
                else:
                    rating += weight_4*score[i-max_ind-1]
            #sharpness
            if rating < 0:
                rating = 0
            
            # Converts the S/N-DM plot into a probability frequency plot
            # Instead of each point in DM space having a corresponding S/N y-value
            # there will be an array containing a number of DM elements proportional to its S/N value
            for i in range(len(signalToDm)):
                temp_arr = []
                frequency = int(signalToDm[i][1]*1000)      # Needs to be an integer, timed it by 1000 to reduce rounding errors, proportions still same
                temp_arr = [signalToDm[i][0]] * frequency   # Creates the corresponding number of elements and adds it to the array
                freq_arr.extend(temp_arr)
            
            
            diff_SN = max(s_meanSN) - (0.5*s_meanSN[0] + 0.5*s_meanSN[-1])
            diff_DM = s_meanDM[-1] - s_meanDM[0] #?????center this around peak
            
            sharp_ratio = diff_SN/(diff_SN + diff_DM) #height/width
            shape_conf = rating/max_score
            
            skewness = skew(freq_arr, axis = 0)
            kurt = kurtosis(freq_arr, axis = 0, fisher = True)
            
            freq_scaled = preprocessing.scale(freq_arr)
            ks_stat = ks_cordes(signalToDm[:,0],signalToDm[:,1],labels_arr[q][:,1],meanDM[max_ind])
            
            shape_vals.append(shape_conf)
            sharp_vals.append(sharp_ratio)
            skew_vals.append(skewness)
            kurt_vals.append(kurt)
            kstest_vals.append(ks_stat)
            cand_paths.append(FILE)
            
            if (counter in pos_array):
                class_vals.append(1)
            else:
                class_vals.append(0)
    

dataframe = pd.DataFrame({'Shape Feature': shape_vals,
                          'Sharp Feature': sharp_vals,
                          'Skewness': skew_vals,
                          'Kurtosis': kurt_vals,
                          'KS-test stat': kstest_vals,
                          'Label': class_vals,
                          'Candidate paths: ': cand_paths})

dataframe.to_csv("feature_table.csv")


         
    
