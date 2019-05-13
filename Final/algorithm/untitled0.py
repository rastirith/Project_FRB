import os, glob
import numpy as np
import pandas as pd

from matrix import matrix_form
from tensorflow.keras.models import load_model

def DF(path):
    """Opens binary encoded file, reshapes it into columns and returns a pandas datafram.

    Keyword arguments:
    path -- the path to the file to be opened
    """
    axislabels = ["DM", "Time", "S/N", "Width", "Label"]     # Labels of data types contained in file
    
    Tfile = open(path,'r')  
    data = np.fromfile(Tfile,np.float32,-1)         # Decodes data to float32 objects
    c = data.reshape((-1,5))                        # Reshapes the string of numbers into columns
    df = pd.DataFrame(c,columns=axislabels)         # Creates dataframe
    Tfile.close()
    
    return df

modelName = "\\utils\\binary_dropout_40_3_conv_64_nodes_1_dense_128_nodes_1557361636"

ddplan = "\\utils\\dd_plan.txt" #Dedispersion Plan Filepath
classifier = load_model(os.getcwd() + modelName)

source_paths = []   # Array of file paths to be reviewed

# Loops through all .dat files to store them in the 'source_paths' array
for file in glob.glob(os.getcwd() + '\\try\\' + "*.dat"):
    source_paths.append(file)

# Loops through the whole file space defined by 'source_paths'
for i in range(len(source_paths)): 
    
    df = DF(source_paths[i]) # Creates dataframe from the .dat file
    X = np.array(df)
    points = X[X[:,0].argsort()]
    #print(points)

    
    matrix = matrix_form(points,ddplan)
    matrix = matrix.reshape(-1,100,100,1) #/0.71 #this mean value
    matrix /= 17.94
    
    #predicting with the CNN
    prediction = classifier.predict_classes(matrix)
    
    print(prediction)
    
"""
matrix = matrix_form(cand,ddplan)
matrix = matrix.reshape(-1,100,100,1) #/0.71 #this mean value
matrix /= 17.94

#matrix = (matrix - scaling[0])/scaling[1]


#predicting with the CNN
prediction = classifier.predict_classes(matrix)"""