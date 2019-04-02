import numpy as np
import pandas as pd
import os
from sklearn import preprocessing
from sklearn.cluster import DBSCAN
from timeit import default_timer as timer

# Creates dataframe for file
def DF(path):
    axislabels = ["DM", "Time", "S/N", "Width", "Label"]
    Tfile = open(path,'r')
    data = np.fromfile(Tfile,np.float32,-1) 
    c = data.reshape((-1,5))
    df = pd.DataFrame(c,columns=axislabels)
    Tfile.close()
    
    return df

def scale(dmMAX, tMAX):
    scaleDUMMY = np.array([[0,0],[0,tMAX],[dmMAX,0],[dmMAX, tMAX]])
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(scaleDUMMY)
    
    return scaler

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


xeps = 0.012
xmin = 3



def cluster(path):
    
    timer7 = []
    start7 = timer()
    
    #fileSize = os.path.getsize(path)/1024000
    df = DF(path) # Creates dataframe from the .dat file
    
    X_db = np.array(df.drop(columns=['Width', 'S/N', 'Label']))  # Drops width and S/N data for DBScan to run on the DM - Time space
    X = np.array(df)
    
    # Sorts the data points by DM
    points_db = X_db[X_db[:,0].argsort()]
    points = X[X[:,0].argsort()]

    # Lower DM limit below which the DBScan is now run as it is most likely not extragalactic
    # Speeds up the DBScan runtime
    dm_lim = 0.03*max(points_db[:,0])
    points_new = np.array(points_db[points_db[:,0] > dm_lim])
    

    X_scaled = scale(1077.4,50.32576).transform(points_new) # Rescales the data so that the x- and y-axes get ratio 1:1
    X_scaled[:,1] = 3*X_scaled[:,1]
    
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

    # Loops through all remaining clusters to exclude further clusters, calculate feature values, and classify them using Random Forest
    labels = np.unique(newArr[:,-1])

    end7 = timer()
    timer7.append(end7 - start7)
    #print("Clustering in module: ", np.mean(timer7))
    
    return newArr, labels
