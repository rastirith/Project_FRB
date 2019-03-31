import numpy as np
import pandas as pd
import matplotlib as mpl
import glob, os, sys
import warnings
import math
import pickle
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn import preprocessing
from matplotlib import pyplot as plt

from featuring import candClassifier

warnings.filterwarnings("ignore", category=mpl.cbook.mplDeprecation)
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.set_printoptions(suppress=True)

def progressBar(value, endvalue, bar_length=20):
    
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    
    sys.stdout.write("\rPercent: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))  
    
# Returns 'num' lowest elements of array 'arr' in a new array
def sort(arr, num):
    xsorted = np.sort(arr)[:num]
    return xsorted

# Re-orders a shuffled labels_arr array so that the index of the array and the
# labels in the labels array has a q-1 relationship counting from q = 0.
# Label -1 (q = 0) corresponds to RFI and everything above it is a legitimate cluster
def clusterOrder(clusterArr):

    lab_arr = np.unique(clusterArr)     # Creates an array containing each unique value of the cluster label array
    for n in range(1, len(lab_arr)):    # Loops through all the possible labels counting from 0 to the number of unique clusters
        clusterArr = np.where(clusterArr == lab_arr[n], n - 1, clusterArr)
    return clusterArr


# Creates dataframe for file
def DF(path):
    axislabels = ["DM", "Time", "S/N", "Width", "Label"]
    Tfile = open(path,'r')
    data = np.fromfile(Tfile,np.float32,-1) 
    c = data.reshape((-1,5))
    df = pd.DataFrame(c,columns=axislabels)
    Tfile.close()
    return df

# Conducts the 2d KS-test on the SN-DM distribution and the theoretical cordes equation
def ks_cordes(dmArr,snArr,timeArr,peakDmMean):
    freq = 0.334
    bandWidth = 64
    cordes = []             # y-values of the theoretical cordes function
    peakSN = max(snArr)     # Value of the higher SN-bin of the data
    snFreqArr = []          # Probability frequency distribution for the data
    cordesFreqArr = []      # Probability frequency distribution for the theoretical function
    
    Wms = np.percentile(timeArr,80)-np.percentile(timeArr,20)   # Time width using the quantile method
    Wms = Wms*1000                                              # Must be in milliseconds
    dmScaled = dmArr - peakDmMean                               # Centers the data around DM = 0
    snRatios = (snArr + 9)/(peakSN + 9)                         # Ratios of the SN-values in relation to the peak
    
    x = np.linspace(min(dmScaled),max(dmScaled),2000)           # X-values for cordes function
    zeta = (6.91*10**-3)*bandWidth*(freq**-3)*(Wms**-1)*x       # Zeta function, see Cordes & M
    
    # Calculates the y-values of the theoretical function
    for i in range(len(x)):
        cordes.append((math.pi**(1/2))*0.5*(zeta[i]**-1)*math.erf(zeta[i]))

    altArr = []

    # Creates prob. freq. representation of the SN distribution
    for i in range(len(snRatios)):
        temp_arr = []
        frequency = int(snRatios[i]*100)      # Needs to be an integer, timed it by 1000 to reduce rounding errors, proportions still same
        altArr.append(frequency)
        temp_arr = [dmScaled[i]] * frequency   # Creates the corresponding number of elements and adds it to the array
        snFreqArr.extend(temp_arr)

    # Creates prob. freq. representation of the cordes func.
    for i in range(len(cordes)):
        temp_arr = []
        frequency = int(cordes[i]*100)
        temp_arr = [x[i]] * frequency
        cordesFreqArr.extend(temp_arr)
        
    statistic = stats.ks_2samp(snFreqArr,cordesFreqArr) #2D KS-test
    return statistic[0]

# Method to calculate and return the theoretical DM range span given a certain
# time duration/width and peak magnitude
def cordes(timeArr,peakSN):
    freq = 0.334
    bandWidth = 64
    cordes = []                 # Array containing all theoretical SN-values from the cordes function
    SNratio = 9/(peakSN + 9)    # Ratio of the peak to the cutoff point in the data
                                # Need to calculate the corresponding DM range for a reduction from 
                                # 1 (peak) to this ratio (tails)
    ratioMargin = 0.05
    if (SNratio - ratioMargin) > 0:
        SNratio -= ratioMargin

    Wms = np.percentile(timeArr,80)-np.percentile(timeArr,20)   # Using the quantile method to calculate the width
    Wms = Wms*1000                                              # Width in ms
    
    x = np.linspace(-1000,1000,10000)   # Generates x-values for the cordes function
    zeta = (6.91*10**-3)*bandWidth*(freq**-3)*(Wms**-1)*x       # Zeta function in the cordes function
    
    first = 0       # Variable indicating whether bottom or top of DM range has been found
    bot_dm = 0      # Value of the lower end of the DM range
    top_dm = 0      # Value of the upper end of the DM range
    for i in range(len(x)): # Loops through the x-values and calculates the cordes value in each step
        y = (math.pi**(1/2))*0.5*(zeta[i]**-1)*math.erf(zeta[i])
        cordes.append(y)
        if (y >= SNratio) and (first == 0): # First time the theoretical ratio goes above the actual ratio go in here
            bot_dm = x[i]                   # This x-value corresponds to the bottom DM value
            first = 1                       # Changes variable value to not go into this statement again
        if (y <= SNratio) and (first == 1): # First time the theoretical ratio goes below the actual ratio after bottom is found
            top_dm = x[i]                   # This x-value corresponds to the top DM value
            break                           # Values have been found, break out of loop
    dm_range = top_dm - bot_dm              # Theoretical allowed DM range for the current candidate
    return dm_range

clf = pickle.load(open("model.sav",'rb'))   # Loads the saved Random Forest model

source_paths = []   # Array of file paths to be reviewed

# Loops through all .dat files to store them in the 'source_paths' array
for file in glob.glob(os.getcwd() + '\\odir\\algTrain4\\' + "*.dat"):
    source_paths.append(file)

"""
try:    # Only creates folders if they don't already exist
    os.mkdir(os.getcwd() + '\idir\\' + "\\candidates")
    os.mkdir(os.getcwd() + '\idir\\' + "\\empty")
except:
    pass

shape_vals = []     # Array containing the shape feature values of the candidates
sharp_vals = []     # Array containing the sharpness feature values of the candidates
skew_vals = []      # Array containing the skewness feature values of the candidates
kurt_vals = []      # Array containing the kurtosis feature values of the candidates
kstest_vals = []    # Array containing the ks-test feature values of the candidates
class_vals = []     # Array containing the classification labels of the candidates"""

TP = 0
TN = 0
FP = 0
FN = 0
#indices = []

totTP = len(source_paths)/2

dmMAX=1077.4
tMAX=50.32576

scaleDUMMY = [[0,0],[0,tMAX],[dmMAX,0],[dmMAX, tMAX]]
scaleDUMMY = np.array(scaleDUMMY)
scaler = preprocessing.MinMaxScaler()
scaler.fit(scaleDUMMY)


# Loops through the whole file space defined by 'source_paths'
for i in range(len(source_paths)): 
    progressBar(i,len(source_paths))
    
    fileSize = os.path.getsize(source_paths[i])/1024000
    path_index = i      # Current path index in the source_paths array
    
    file = source_paths[path_index]     # Setting which file to open
    df = DF(file) # Creates dataframe from the .dat file
    orig_X = np.array(df)   # Sets dataframe as a Numpy array
    
    X_db = np.array(df.drop(columns=['Width', 'S/N', 'Label']))  # Drops width and S/N data for DBScan to run on the DM - Time space
    X = np.array(df)
    
    # Sorts the data points by DM
    points_db = X_db[X_db[:,0].argsort()]
    points = X[X[:,0].argsort()]

    # Lower DM limit below which the DBScan is now run as it is most likely not extragalactic
    # Speeds up the DBScan runtime
    dm_lim = 0.03*max(points_db[:,0])
    points_new = np.array(points_db[points_db[:,0] > dm_lim])
    
    X_scaled = scaler.transform(points_new) # Rescales the data so that the x- and y-axes get ratio 1:1
    X_scaled[:,1] = 3*X_scaled[:,1]
    
    """
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax3.scatter(X_scaled[:,1], X_scaled[:,0], s = 3)
    ax3.set_xlabel(file.split("odir\\")[1])
    ax3.set_xlim(0,3)
    ax3.set_ylim(0,1)"""
    
    xeps = 0.012     # Radius of circle to look around for additional core points
    xmin = 3         # Number of points within xeps for the point to count as core point
    
    clusters = DBSCAN(eps=xeps, min_samples = xmin).fit_predict(X_scaled)   # Clustering algorithm, returns array with cluster labels for each point

    # Re-inserts bottom points with labels -1 for RFI
    length = len(points) - len(clusters)
    clusters = np.insert(clusters,0,np.full(length,-1))

    # Adds column to the points arrays for cluster label
    newArr = np.column_stack((points, clusters[np.newaxis].T))

    # Re-order
    newArr[:,-1] = clusterOrder(newArr[:,-1])
    
    # Noise condition for Nevents<Nmin => noise
    N_min = 20 # Tuneable number (set to match Karako)
    labels = np.unique(newArr[:,-1])
    for q in range(1,len(labels)):
        label = labels[q]
        labSlice = np.extract(newArr[:,-1] == label, newArr[:,-1])
        if (len(labSlice) < N_min):        # Gives points of clusters with less than N_min events the RFI lable of -1
            newArr[:,-1] = np.where(newArr[:,-1] == label, -1, newArr[:,-1])
    
    # Re-order    
    newArr[:,-1] = clusterOrder(newArr[:,-1])
    
    # Break
    dm_lim = 40         # Increased DM-limit to check for clusters with 'fraction' of their points below this limit
    fraction = 0.05     # Size of the fraction of points allowed in below 'dm_lim'

    # Condition that sets all clusters with 'fraction' of its points below dm_lim to also be classified as RFI
    labels = np.unique(newArr[:,-1])
    for q in range(1,len(labels)):
        label = labels[q]
        labSlice = np.extract(newArr[:,-1] == label, newArr[:,-1])
        num_temp = int(round(fraction*len(labSlice),0))    # Calculates how many points of a label a certain fraction corresponds to and rounds to nearest integer
        temp = sort(newArr[newArr[:,-1] == label][:,0],num_temp)                # Returns the 'num_temp' lowest dms in labels_arr[q]
        if ((len(temp) > 0) and (max(temp) < dm_lim)):          # If the highest number in temp is below dm_lim then so is the rest in 'temp'
            newArr[:,-1] = np.where(newArr[:,-1] == label, -1, newArr[:,-1])

    # Condition that sets all points with dm below dm_lim to be classified as RFI
    newArr[:,-1][newArr[:,0] < dm_lim] = -1

    # Re-order    
    newArr[:,-1] = clusterOrder(newArr[:,-1])

    # Burst duration condition
    labels = np.unique(newArr[:,-1])
    for q in range(1,len(labels)):
        label = labels[q]
        upper = np.quantile(newArr[newArr[:,-1] == label][:,1], 0.8)
        lower = np.quantile(newArr[newArr[:,-1] == label][:,1], 0.2)
        if (upper - lower) >= 1:            # If the time between the quantiles is longer than 1s set cluster to RFI
            newArr[:,-1] = np.where(newArr[:,-1] == label, -1, newArr[:,-1])

    # Re-order    
    newArr[:,-1] = clusterOrder(newArr[:,-1])
    
    least_acc = []      # Contains the points of all candidates classified as 'least acceptable'
    good = []           # Contains the points of all candidates classified as 'good'
    excellent = []      # Contains the points of all candidates classified as 'excellent'
    labelNumber = 0     # Label number of clusters in same files, used to separate multiple clusters of equal classification

    # Loops through all remaining clusters to exclude further clusters, calculate feature values, and classify them using Random Forest
    labels = np.unique(newArr[:,-1])

    for q in range(1,len(labels)):
        #break
        label = labels[q]
        result = candClassifier(label, newArr)
        
        dmData = newArr[newArr[:,-1] == label][:,0]
        timeData = newArr[newArr[:,-1] == label][:,1]
        snData = newArr[newArr[:,-1] == label][:,2]
        labData = newArr[newArr[:,-1] == label][:,4]
        
        if np.mean(labData) > 0.15 and result[0] == 1:
            TP += 1
        elif np.mean(labData) > 0.15 and result[0] == 0:
            FN += 1
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(newArr[:,1], newArr[:,0], s = 4)
            ax.scatter(timeData,dmData, s = 4, color = 'r')
            
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(111)
            ax1.scatter(dmData, snData, s = 4)
        elif np.mean(labData) < 0.15 and result[0] == 1:
            FP += 1
        elif np.mean(labData) < 0.15 and result[0] == 0:
            TN += 1



progressBar(1,1)
print("Recall: " + str(round((TP/(TP + FN)),2)))
print("PPV: " + str(round((TP/(TP + FP)),2)))
print("Fraction of planted cands: " + str(TP/totTP))
#print("TP Indices: " + str(indices))
print("TP: " + str(TP))
print("TN: " + str(TN))
print("FP: " + str(FP))
print("FN: " + str(FN))









"""
    # Re-order    
    newArr[:,4] = clusterOrder(newArr[:,4])

    least_acc = np.array(least_acc)
    good = np.array(good)
    excellent = np.array(excellent)
    
    total = []
    if_var = 0
    
    # Adds a column to the numpy array with an integer corresponding to the cluster classification
    # 3 = excellent, 2 = good, 1 = least acceptable
    if (len(excellent) > 0):
        excellent_c = np.full((len(excellent),7), 3, dtype = float)
        excellent_c[:,:-1] = excellent
        total.extend(excellent_c)
        if_var += 1
    if (len(good) > 0):
        good_c = np.full((len(good),7), 2, dtype = float)
        good_c[:,:-1] = good
        total.extend(good_c)
        if_var += 1
    if (len(least_acc) > 0):
        least_acc_c = np.full((len(least_acc),7), 1, dtype = float)
        least_acc_c[:,:-1] = least_acc
        total.extend(least_acc_c)
        if_var += 1
        
    total = np.array(total)         # Sets the total array containing feature values, class labels, and cluster numbers to Numpy array
    
    if len(total) > 0:              # If the 'total' array contains candidates then go in here
        dataframe = pd.DataFrame({'DM': total[:,0],         # Creates a dataframe with feature values, class labels, and cluster numbers
                                  'Time': total[:,1],
                                  'S/N': total[:,2],
                                  'Width': total[:,3],
                                  'Class': total[:,6],
                                  'Cluster Number': total[:,5]})
        
        new_name = source_paths[path_index].split("idir\\")[1].replace('.dat','_c.csv')               # Name of the class file to be created
        dataframe.to_csv(os.getcwd() + '\\idir\\' + "\\candidates\\" + new_name, index = False) # Saves csv file of the above dataframe
        new_path = os.getcwd() + '\\idir\\' + "\\candidates\\" + source_paths[path_index].split("idir\\")[1]  # Path to the created file
        
        if (os.path.isfile(new_path)) == False:     # Moves the .dat file to the 'candidates' directory only if there isn't already an
            os.rename(file, new_path)               # identical .dat file there.
            
    else:                           # If there are no candidates then move the .dat file to the 'empty' directory
        new_path = os.getcwd() + '\\idir\\' + "\\empty\\" + source_paths[path_index].split("idir\\")[1]
        if (os.path.isfile(new_path)) == False:
            os.rename(file, new_path)"""

         