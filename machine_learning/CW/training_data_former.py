#Formating Training Data
#reshape the constructed matrix files for use in CNN 
#saving with pickle

import numpy as np
import os
import random
import pickle
from sklearn import preprocessing

DATADIR = os.getcwd() + "\matrix_files\\Playground3\\"

arrLength = 40000
training_data = []

CATEGORIES=["Noise","Burst"]

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR,category)
        class_num = CATEGORIES.index(category)
        for matrix in os.listdir(path):
            if len(training_data)==arrLength:
                break
            try:
                matrix_arr = np.load(os.path.join(path,matrix))
                training_data.append(np.array([matrix_arr, class_num]))
            except:
                pass
       
create_training_data()      

random.shuffle(training_data)
training_data = np.array(training_data)

X = []
y = []
"""
c = np.arange(arrLength)

Xalt = np.empty(arrLength, dtype = np.float64)
yalt = np.empty(arrLength, dtype = training_data[:,1].dtype)

print(Xalt.dtype)
Xalt[c] = training_data[:,0][c].astype(np.float64)
yalt[c] = training_data[:,1][c]"""

#print(Xalt[c])
for features, label in training_data:
    X.append(features)
    y.append(label)
    #print(features)
"""
print(np.array(X).dtype)
print("\n")
print(Xalt.dtype)"""
#print(np.array(X))
X = np.array(X).reshape(-1, 70, 100, 1) #download more ram or batch it 
X = X/np.amax(X)


"""
scaler = preprocessing.MinMaxScaler()
X_scaled = scaler.fit_transform(X)"""

pickle_out = open("X_scaled.pickle","wb")
pickle.dump(X,pickle_out)
pickle_out.close()

pickle_out = open("y2.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()


