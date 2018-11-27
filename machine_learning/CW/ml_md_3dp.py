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

for i in range(44,45):
    
    FILE = source_paths[i]
    #getting df for test file
    df = DF(FILE)
    
    print(df.head())
    
    #preprocesssing
    orig_X = np.array(df)
    X = np.array(df.drop(columns=['Width','S/N']))
    print(X)
    #scale_X = preprocessing(X)
    scale_X = StandardScaler().fit_transform(X)
    print(len(scale_X))
    
    
    #fittitng model
    """
    clf = MeanShift()
    clf.fit(scale_X)
    """
    xeps = 0.05*max(scale_X[:,0])
    xmin = 5
    clf = DBSCAN(eps=xeps, min_samples=xmin,n_jobs=-1,metric='minkowski',p=2)
    clf.fit(scale_X)
    #labels = db.fit_predict(x)
    #intereting stuff
    labels = clf.labels_
    #cluster_centers = clf.cluster_centers_
    #n_clusters = len(np.unique(labels))
    
    print(np.unique(labels))
    """
    #adding group to data frame
    df['cluster_group'] = np.nan
    for i in range(len(X)):
        df['cluster_group'].iloc[i] = labels[i]
    """
    
    #plotting
    fig = plt.figure()
    plt.scatter(X[:, 1],X[:, 0], c=labels[:], cmap="brg", alpha=0.4)
    #fig.set_xlabel('Time')
    #fig.set_ylabel('DM')
    plt.title(FILE)
    plt.show()
    """
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(orig_X[:, 1], orig_X[:, 0],orig_X[:,2],c=labels[:], cmap="brg", alpha=0.4)
    plt.title(FILE)
    plt.show()
    """
