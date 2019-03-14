#looking at data file
#Finding the max DM and time in the dataset

import numpy as np
import pandas as pd
import glob, os

#fills data frame
def DF(path):
    axislabels = ["DM", "Time", "S/N", "Width"]
    Tfile = open(path,'r')
    data = np.fromfile(Tfile,np.float32,-1) 
    c = data.reshape((-1,4))
    df = pd.DataFrame(c,columns=axislabels)
    Tfile.close()
    return df

#array of file locations and chosing the file to inspect with path
source_paths = []

#filling source_paths from the idir
for file in glob.glob(os.getcwd() + '\idir\\' + "*.dat"):
    source_paths.append(file)

width_max =[]
w_u = []
for i in range(0,72):    
    path=i
    print(i)
    #setting which file to open
    FILE = source_paths[path]
    #getting df for test file
    df = DF(FILE)
    """print(df)"""
    a=np.array(df["Time"])
    b=np.unique(a)
    for i in range(len(b)):
        w_u.append(b[i])
    width_max.append(df["Time"].max())
print("check")
print(max(np.unique(width_max)))
