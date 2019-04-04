#Formating Training Data
#reshape the constructed matrix files for use in CNN 
#saving with pickle
import numpy as np
import os
import random
import pickle

DATADIR = os.getcwd() + "\matrix_files\\Playground3\\"

arrLength = 5

CATEGORIES=["Noise","Burst"]

def create_training_data():
    matrix_data = []
    label_data = []
    for category in CATEGORIES:
        path = os.path.join(DATADIR,category)
        class_num = CATEGORIES.index(category)
        for matrix in os.listdir(path):
            
            if len(matrix_data)==arrLength:
                break
            try:
                matrix_data.append(np.load(os.path.join(path,matrix)))
                label_data.append(class_num)
            except:
                pass   
    
    random.seed(6)
    random.shuffle(matrix_data)
    random.seed(6)
    random.shuffle(label_data)
    
    return np.array(matrix_data), np.array(label_data)
          
data = create_training_data()   

X = data[0]
y = data[1]

X = X.reshape(-1, 70, 100, 1) #download more ram or batch it 
X = X/np.amax(X)

pickle_out = open("X_scaled.pickle","wb")
pickle.dump(X,pickle_out)
pickle_out.close()

pickle_out = open("y2.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()


