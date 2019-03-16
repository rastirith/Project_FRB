import numpy as np
import pandas as pd
import matplotlib as mpl
import glob, os
import warnings
import math
import pickle
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn import preprocessing
from scipy.stats import skew, kurtosis
import sys
from timeit import default_timer as timer


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
    
    Wms = np.percentile(timeArr,75)-np.percentile(timeArr,25)   # Time width using the quantile method
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

    Wms = np.percentile(timeArr,75)-np.percentile(timeArr,25)   # Using the quantile method to calculate the width
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
for file in glob.glob(os.getcwd() + '\idir\\input\\' + "*.dat"):
    source_paths.append(file)

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
class_vals = []     # Array containing the classification labels of the candidates


# Loops through the whole file space defined by 'source_paths'
for i in range(2000,20000): 
    start = timer()
    progressBar(i, len(source_paths))
    end = timer()
    print(end-start)
    fileSize = os.path.getsize(source_paths[i])/1024000
    path_index = i      # Current path index in the source_paths array
    break
    file = source_paths[path_index]     # Setting which file to open
    
    df = DF(file)                       # Creates dataframe from the .dat file
    #print(source_paths)
    X_db = np.array(df.drop(columns=['Width', 'S/N', 'Label']))  # Drops width and S/N data for DBScan to run on the DM - Time space
    X = np.array(df)
    
    # Sorts the data points by DM
    points_db = X_db[X_db[:,0].argsort()]
    points = X[X[:,0].argsort()]

    # Lower DM limit below which the DBScan is now run as it is most likely not extragalactic
    # Speeds up the DBScan runtime
    dm_lim = 0.03*max(points_db[:,0])
    points_new = np.array(points_db[points_db[:,0] > dm_lim])
    
    X_scaled = preprocessing.MinMaxScaler().fit_transform(points_new) # Rescales the data so that the x- and y-axes get ratio 1:1
    
    xeps = 0.025     # Radius of circle to look around for additional core points
    xmin = 3         # Number of points within xeps for the point to count as core point
    
    clusters = DBSCAN(eps=xeps, min_samples = xmin).fit_predict(X_scaled)   # Clustering algorithm, returns array with cluster labels for each point

    # Re-inserts bottom points with labels -1 for RFI
    length = len(points) - len(clusters)
    clusters = np.insert(clusters,0,np.full(length,-1))
    
    # Adds column to the points arrays for cluster label
    newArr = np.column_stack((points, clusters[np.newaxis].T))
    
    # Re-order
    newArr[:,4] = clusterOrder(newArr[:,4])
    
    # Noise condition for Nevents<Nmin => noise
    N_min = 20 # Tuneable number (set to match Karako)
    labels = np.unique(newArr[:,4])
    for q in range(1,len(labels)):
        label = labels[q]
        labSlice = np.extract(newArr[:,4] == label, newArr[:,4])
        if (len(labSlice) < N_min):        # Gives points of clusters with less than N_min events the RFI lable of -1
            newArr[:,4] = np.where(newArr[:,4] == label, -1, newArr[:,4])
    
    # Re-order    
    newArr[:,4] = clusterOrder(newArr[:,4])
    
    # Break
    dm_lim = 40         # Increased DM-limit to check for clusters with 'fraction' of their points below this limit
    fraction = 0.05     # Size of the fraction of points allowed in below 'dm_lim'

    # Condition that sets all clusters with 'fraction' of its points below dm_lim to also be classified as RFI
    labels = np.unique(newArr[:,4])
    for q in range(1,len(labels)):
        label = labels[q]
        labSlice = np.extract(newArr[:,4] == label, newArr[:,4])
        num_temp = int(round(fraction*len(labSlice),0))    # Calculates how many points of a label a certain fraction corresponds to and rounds to nearest integer
        temp = sort(newArr[newArr[:,4] == label][:,0],num_temp)                # Returns the 'num_temp' lowest dms in labels_arr[q]
        if ((len(temp) > 0) and (max(temp) < dm_lim)):          # If the highest number in temp is below dm_lim then so is the rest in 'temp'
            newArr[:,4] = np.where(newArr[:,4] == label, -1, newArr[:,4])

    # Condition that sets all points with dm below dm_lim to be classified as RFI
    newArr[:,4][newArr[:,0] < dm_lim] = -1

    # Re-order    
    newArr[:,4] = clusterOrder(newArr[:,4])

    # Burst duration condition
    labels = np.unique(newArr[:,4])
    for q in range(1,len(labels)):
        label = labels[q]
        upper = np.quantile(newArr[newArr[:,4] == label][:,1], 0.8)
        lower = np.quantile(newArr[newArr[:,4] == label][:,1], 0.2)
        if (upper - lower) >= 1:            # If the time between the quantiles is longer than 1s set cluster to RFI
            newArr[:,4] = np.where(newArr[:,4] == label, -1, newArr[:,4])

    # Re-order    
    newArr[:,4] = clusterOrder(newArr[:,4])
    
    least_acc = []      # Contains the points of all candidates classified as 'least acceptable'
    good = []           # Contains the points of all candidates classified as 'good'
    excellent = []      # Contains the points of all candidates classified as 'excellent'
    labelNumber = 0     # Label number of clusters in same files, used to separate multiple clusters of equal classification

    # Loops through all remaining clusters to exclude further clusters, calculate feature values, and classify them using Random Forest
    labels = np.unique(newArr[:,4])
    for q in range(1,len(labels)):
    
        label = labels[q]
        dmData = newArr[newArr[:,4] == label][:,0]
        timeData = newArr[newArr[:,4] == label][:,1]
        snData = newArr[newArr[:,4] == label][:,2]
        
        signalToDm = list(zip(dmData, snData))  # Array containing DM - SN data
        signalToDm = np.array(signalToDm)
        min_val = min(signalToDm[:,1])                                  # Finds the lowest SN-value in the candidate array

        scaled_signal = preprocessing.MinMaxScaler().fit_transform(signalToDm)  # Preprocesses signalToDM to used for sharpness condition

        # Sets y = 0 for visualisation
        for i in range(len(signalToDm[:,1])):
            signalToDm[:,1][i] = signalToDm[:,1][i] - min_val
            
        # Splitting into chunks of equal number of events in each
        split_param = 7 # Number of chunks to be split into
        dummy = np.array_split(signalToDm,split_param)
        
        meanSN = []     # Contains mean SN value of each chunk 
        meanDM = []     # Contains mean DM value of each chunk
        
        # Loops through the chunks, calculates the relevant mean values and puts them into the appropriate arrays
        for i in range(len(dummy)):
            tempSN = np.mean(dummy[i][:,1])
            tempDM = np.mean(dummy[i][:,0])
            meanSN.append(tempSN)
            meanDM.append(tempDM)
        
        max_val = max(meanSN + min_val) # Array containing the real (not reduced) peak SN values in each chunk
        max_ind = np.argmax(meanSN)     # Finds the index for the highest SN bin value
        peakMeanDm = meanDM[max_ind]    # The corresponding DM for this bin value
        
        upper_dm_range = cordes(timeData, meanSN[max_ind]) # The theoretically widest allowed DM range of candidate
        dm_diff = max(dmData) - min(dmData)  # The DM range spanned by the actual data

        # Condition that excludes the candidates with SN peaks too far from the centre, as well as candidates spanning a range that is too wide
        # Using 2 times the upper_dm_range to use a large safety margin
        if (max_ind > 4) or (max_ind < 2) or (dm_diff > 2*upper_dm_range):  
            newArr[:,4] = np.where(newArr[:,4] == label, -1, newArr[:,4])
    
        # All candidates that haven't been excluded thus far go in here
        else:
            counter += 1        # Final candidate number reaching this point
            freq_arr = []       # Probability frequency distribution representation of the DM - SN data
            weight_1 = -1       # Score weight if ratio is less than 1 but more than 1 - check_1
            weight_2 = -0.3     # Score weight if ratio is less than 1 - check_1, but more than 1 - check_2
            weight_3 = 1        # Score weight if ratio is less than 1 - check_2
            weight_4 = -1       # Score weight if ratio is more than 1
            check_1 = 0.075
            check_2 = 0.15
            score = [0,1.3,2.5,2.5]                         # Scoring system where one index step corresponds to one step from peak bin
            max_score = 2*(score[0] + score[1] + score[2])  # Maximum possible score
            rating = 0  # Rating score after weight and scores have been applied
            
            for i in range(max_ind - 1, -1, -1):                # Loops through all bins from the peak bin moving to the left
                ratio=meanSN[i]/meanSN[i+1]                     # Ratio of the next bin to the previous bin
            
                if ((ratio>=(1-check_1)) and (ratio<=1)):
                    rating += weight_1*score[max_ind-(i+1)]
                elif ((ratio>=(1-check_2)) and (ratio<=1)):
                    rating += weight_2*score[max_ind-(i+1)]
                elif (ratio<=1):
                    rating += weight_3*score[max_ind-(i+1)]
                else:
                    rating += weight_4*score[max_ind-(i+1)]
                            
            for i in range((max_ind+1),split_param):            # Loops through all bins from the peak bin moving to the right
                ratio=meanSN[i]/meanSN[i-1]

                if ((ratio>=(1-check_1)) and (ratio<=1)):
                    rating += weight_1*score[i-max_ind-1]
                elif ((ratio>=(1-check_2)) and (ratio<=1)):
                    rating += weight_2*score[i-max_ind-1]
                elif ratio <=1:
                    rating += weight_3*score[i-max_ind-1]
                else:
                    rating += weight_4*score[i-max_ind-1]
                    
            # Exception case where rating is less than 0, sets rating to 0 if this happens
            if rating < 0:
                rating = 0

            # Converts the S/N-DM plot into a probability frequency plot
            # Instead of each point in DM space having a corresponding S/N y-value
            # there will be an array containing a number of DM elements proportional to its S/N value
            normal_snRatios = (signalToDm[:,1] + 9)/(max(signalToDm[:,1]) + 9)
            for i in range(len(signalToDm)):
                temp_arr = []
                frequency = int((normal_snRatios[i])*1000)      # Needs to be an integer, timed it by 1000 to reduce rounding errors, proportions still same
                temp_arr = [signalToDm[i][0]] * frequency   # Creates the corresponding number of elements and adds it to the array
                freq_arr.extend(temp_arr)

            # FEATURES
            shape_conf = rating/max_score                       # Shafe conf feature
            skewness = skew(freq_arr, axis = 0)                 # Skewness feature
            kurt = kurtosis(freq_arr, axis = 0, fisher = True)  # Kurtosis feature
            ks_stat = ks_cordes(signalToDm[:,0],signalToDm[:,1],timeData,meanDM[max_ind])     # KS feature
            
            # Adds the feature values to the corresponding arrays
            shape_vals.append(shape_conf)
            skew_vals.append(skewness)
            kurt_vals.append(kurt)
            kstest_vals.append(ks_stat)
            
            # Saves the classification label of the cluster to the class_vals array
            # 1 = candidate, 0 = no candidate
            if (counter in pos_array):
                class_vals.append(1)
            else:
                class_vals.append(0)


# Creates dataframe table containing all feature values as well as the classification labels for each cluster
dataframe = pd.DataFrame({'Shape Feature': shape_vals,
                          'Skewness': skew_vals,
                          'Kurtosis': kurt_vals,
                          'KS-test stat': kstest_vals,
                          'Label': class_vals})

dataframe.to_csv("feature_table.csv")   # Writes dataframe to .csv file


         
    
