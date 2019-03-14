#crab candidate tester

import pandas as pd
import numpy as np
import glob,os
from matplotlib import pyplot as plt
from sklearn import preprocessing
from timeit import default_timer as timer


def DF_crab(path):
    labels = ["Time", "DM", "Width", "S/N"]
    #reading in data files    
    df = pd.read_csv(source_paths[x], delimiter="       ", header = 0, names = labels, engine='python')
    df["Time"] *=1/1000
    df = df[["DM", "Time", "S/N", "Width"]]
    return df


#array of file locations and chosing the file to inspect with path
source_paths = []

#filling source_paths from the idir
for file in glob.glob(os.getcwd() + '\crab_cands\\' + "*.spccl"):
    source_paths.append(file)


start = timer()
dmMAX=1634.5
tMAX=13.4701 

scaleDUMMY = [[0,0],[0,tMAX],[dmMAX,0],[dmMAX, tMAX]]
scaleDUMMY = np.array(scaleDUMMY)
scaler = preprocessing.MinMaxScaler()
scaler.fit(scaleDUMMY)

end = timer()
print(end-start)
for x in range(2,len(source_paths)-1):      
    df=DF_crab(source_paths[x])         
    #print(df)
    
    
    X_db = np.array(df.drop(columns=['Width', 'S/N']))

    #print(X_db[:5])
    #print(scaleDUMMY)
    df.plot(x="Time", y="DM", kind="scatter", title=x)
    start = timer()
    x_scaled = scaler.transform(X_db)
    end = timer()
    print(len(x_scaled)," ",end-start)
    DUMMY_scaled = scaler.transform(scaleDUMMY)
    
    #print(x_scaled[:5])
    #print(DUMMY_scaled)
    
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.set_xlim(0,1)
    ax1.set_ylim(0,1)
    ax1.scatter(x_scaled[:,1], x_scaled[:,0], s = 6, color = '0.7', vmin = -1)
    #ax1.scatter(timeData, dmData, s = 6, color = r, vmin = -1)
    ax1.set_title(str(x))
    #ax1.set_xlabel(str(timeDiff))
    break
    plt.show()
    