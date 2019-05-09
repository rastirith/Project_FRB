#Formating Training Data
#reshape the constructed matrix files for use in CNN 
#saving with pickle
import numpy as np
import os
import random
import pickle

DATADIR = os.getcwd() + "\matrix_files\\val_test\\"

#arrLength = 5

CATEGORIES=["Noise","Burst"]

def create_training_data():
    matrix_data = []
    label_data = []
    for category in CATEGORIES:
        path = os.path.join(DATADIR,category)
        class_num = CATEGORIES.index(category)
        for matrix in os.listdir(path):
            
            #if len(matrix_data)==1:
                #break
            try:
                matrix_data.append(np.load(os.path.join(path,matrix)))
                label_data.append(class_num)
            except:
                pass   
            #matrix_data[-1] /= np.amax(matrix_data[-1])
            
    random.seed(6)
    random.shuffle(matrix_data)
    random.seed(6)
    random.shuffle(label_data)
    
    return np.array(matrix_data), np.array(label_data)
          
data = create_training_data()   

X = data[0]
y = data[1]

X = X.reshape(-1, 100, 100, 2) #last digit 1 for just SN, 2 for SN & width


mean = np.mean(X)#[np.nonzero(X)])
sDev = np.std(X)#[np.nonzero(X)])

X = (X - mean)/sDev
#X = X/np.mean(X)
scaling = [mean, sDev]
scaling = np.array(scaling)
print(scaling)
"""
Scaler = NDStandardScaler()
Scaling = Scaler.fit(X)
X = Scaling.transform(X)

print(Scaling)
"""
#download more ram or batch it 
#X = X/np.amax(X)
#print(np.unique(X[0]))

pickle_out = open("X_scaled.pickle","wb")
pickle.dump(X,pickle_out)
pickle_out.close()

pickle_out = open("y2.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_out = open("scale_param.pickle","wb")
pickle.dump(scaling,pickle_out)
pickle_out.close()
