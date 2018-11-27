import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import DBSCAN
from matplotlib.figure import Figure
import glob, os


"""
n = np.random.normal(size=10000,loc = 0, scale = 1)
result = stats.kstest(n, "norm")
print(result)
plt.hist(n, bins = 100)"""
def val(path, ref):
    
    #Imports data from the .dat file located at the specified path
    Tfile = open(path,'r')
    data = np.fromfile(Tfile,np.float32,-1) 
    c = data.reshape((-1,4))
    Tfile.close()
    refarr = []
    #Defines labels for the axes corresponding to the four different sets of data available
    axislabels = ["DM", "Time", "StoN", "Width"]
    columns = np.hsplit(c,4) #x- and yref values dm=0, time=1, ston=2, width=3
    for i in range(len(columns[ref])):
        refarr.append(columns[ref][i][0])
    #print(columns[0][0][0])
    return refarr

source_paths = []

for file in glob.glob(os.getcwd() + '\idir\\' + "*.dat"):
    source_paths.append(file)
    


#fig = plt.figure()
#ax = Axes3D(fig)
x = val(source_paths[45],0)
y = val(source_paths[45],1)
#for i in range(len(x)):
#    print(x[i],y[i])
    


points = list(zip(x, y))
#print(points)
#z = val(source_paths[0],2)
db = DBSCAN(eps=1, min_samples=5).fit(points)
labels = db.fit_predict(points)
print(set(labels))
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
progress = 0
#print(n_clusters_)
for i in range(len(labels)):
    
    if (i/len(labels)) >= progress:
        print(str(100*i/len(labels)) + "%")
        progress += 0.05
    
    if labels[i] == 0:
        #print(x[i])
        plt.plot(y[i],x[i],"rx")
    elif labels[i] == 1:
        #print(x[i])
        plt.plot(y[i],x[i],"bx")
    elif labels[i] == 2:
        #print(x[i])
        plt.plot(y[i],x[i],"yx")
    elif labels[i] == 3:
       # print(x[i])
        plt.plot(y[i],x[i],"gx")
    #elif labels[i] == -1:
        #print(x[i])
        #plt.plot(y[i],x[i],"co")
    else:
        plt.plot(y[i],x[i],"cx",markersize = 1)
        
    
   
#plt.show()
#ax.set_xlabel("DM")
#ax.set_ylabel("TIME")
#plt.plot(y,x,"ro",markersize = 1.5)

#plt.show()
"""

x = []
y = []
m = []
xlabels = []

for i in range(150):
    x.append(np.random.random() + 1.5)
    y.append(np.random.random() + 1)
    m.append([(np.random.random() + 1.5), (np.random.random() + 1)])
    xlabels.append(0)
for i in range(150):
    x.append(np.random.random())
    y.append(np.random.random())
    m.append([(np.random.random()), (np.random.random())])
    xlabels.append(0)
for i in range(100):
    x.append(np.random.random()*0.5 + 2)
    y.append(np.random.random()*0.5 - 0.5)
    m.append([(np.random.random()*0.5 + 2), (np.random.random()*0.5 - 0.5)])
    xlabels.append(0)
for i in range(300):
    x.append(np.random.random()*5 - 1)
    y.append(np.random.random()*5 - 1)
    m.append([(np.random.random()*5 - 1), (np.random.random()*5 - 1)])
    xlabels.append(-1)

n_clusters = 3
clf = KMeans(n_clusters = 3)
clf.fit(m)
#centers = clf.cluster_centers_
#labels = clf.predict(m)

ms = MeanShift()
ms.fit(m)
centers = ms.cluster_centers_
labels = ms.labels_"""
"""
eps_step = 0.01
samp_step = 1
prev_move = (0,0)
new_samp = 100
new_eps = 0.15
for i in range(5):
    db = DBSCAN(eps=new_eps, min_samples=new_samp).fit(m)
    labels = db.fit_predict(m)
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    #plt.plot(x,y,"bo")
    #plt.plot(clustering,"ro")
    
    correct = 0
    
    for i in range(len(labels)):
        if (labels[i] == -1):
            #plt.plot(x[i],y[i],"bo")
            if (xlabels[i] == -1):
                correct += 1
        elif(labels[i] == 0):
            #plt.plot(x[i],y[i],"yo")
            if (xlabels[i] >= 0):
                correct += 1
        elif(labels[i] == 1):
            #plt.plot(x[i],y[i],"ro")
            if (xlabels[i] >= 0):
                correct += 1
        elif(labels[i] == 2):
            #plt.plot(x[i],y[i],"go")
            if (xlabels[i] >= 0):
                correct += 1
        #plt.plot(x[i],y[i],"ro")
    
    precision = (correct/len(labels))
    
    #plt.plot(x,y,"rx")
    print("Number of clusters: " + str(n_clusters_))
    print("Precision: " + str(round((precision*100),1)) + "%")
    
    if (prev_move == (0,0)):
        new_eps -= eps_step
        prev_move = (-1,0)
    elif(prev_move == (-1,0)):
        new_eps += eps_step
        prev_move = (1,0)
    elif(prev_move == (1,0)):
        new_samp -= samp_step
        prev_move = (0,-1)
    elif(prev_move == (0,-1)):
        new_samp += samp_step
        prev_move = (0,1)
    elif(prev_move == (0,1)):
        new_eps -= eps_step
        new_samp += samp_step
        prev_move = (0,-1)
            
#print(labels)
"""

