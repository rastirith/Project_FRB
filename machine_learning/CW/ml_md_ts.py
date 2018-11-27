import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import DBSCAN
from matplotlib.figure import Figure
import glob, os
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from math import ceil

#array of .dat file paths
source_paths = []
for file in glob.glob(os.getcwd() + '\idir\\' + "*.dat"):
    source_paths.append(file)

#creates dataframe for file
def DF(path):
    axislabels = ["DM", "Time", "S/N", "Width"]
    Tfile = open(path,'r')
    data = np.fromfile(Tfile,np.float32,-1) 
    c = data.reshape((-1,4))
    df = pd.DataFrame(c,columns=axislabels)
    Tfile.close()
    return df


for i in range(42,50):
        
    FILE = source_paths[i]
    #getting df for test file
    df = DF(FILE)
    orig_X = np.array(df)
    X = np.array(df.drop(columns=['Width']))#,'S/N']))
      
    X = X[X[:,1].argsort()]
    
    up = ceil(X[-1][1])
    low_lim = 0
    
    fig = plt.figure() 
    #ax = fig.add_subplot(111,projection='3d')
    
    for k in range(1,up + 1):
        temp_arr = []
        
        for i in range(low_lim,len(X)):
            temp_arr.append(X[i])
           
            if (X[i][1] >= k):
                low_lim = i
                del temp_arr[-1]
                
                if (len(temp_arr) > 1):
                    
                    temp_arr = np.array(temp_arr)
                    
                    X_scaled = StandardScaler().fit_transform(temp_arr)
                    #print(X_scaled)
                    # cluster the data into five clusters
                    xeps = 0.1*max(X_scaled[:,0])
                    xmin = 5
    
                    clusters = DBSCAN(eps=xeps, min_samples = xmin).fit_predict(X_scaled)
                    
                    """
                    for m in range(len(np.unique(clusters))):
                        temp_lab = []
                        #print(m + 1)
                        for p in range(len(clusters)):
                            if (clusters[p] == m):
                               temp_lab.append(X_scaled[p])
                               
                        
                        temp_lab = np.array(temp_lab)
                        print(len(temp_lab))
                        #print(np.argpartition(temp_lab, 5))
                    """
                    #print(clusters)
                    
                    plt.scatter(temp_arr[:, 1], temp_arr[:, 0], c=clusters, cmap="brg",vmin=-1, alpha = 0.4)
                    #ax.scatter(temp_arr[:, 1], temp_arr[:, 0],temp_arr[:,2],c=clusters, cmap="brg", alpha=0.4)
                break
    
    if (len(temp_arr) > 0):
        temp_arr = np.array(temp_arr)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(temp_arr)
        
        xeps = 0.1*max(X_scaled[:,0])
        xmin = 5
        clusters = DBSCAN(eps=xeps, min_samples = xmin).fit_predict(X_scaled)
        
        plt.scatter(temp_arr[:, 1], temp_arr[:, 0], c=clusters, cmap="brg", alpha = 0.4)    
        #ax.scatter(temp_arr[:, 1], temp_arr[:, 0],temp_arr[:,2],c=clusters, cmap="brg", alpha=0.4)
    
    
    
    plt.title(FILE)
    plt.show()
    print("Time steps: " + str(up))