import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import glob, os, sys
import warnings


from clustering import cluster
from matrix import matrix_form
from tensorflow.keras.models import load_model

from timeit import default_timer as timer


def progressBar(value, endvalue, bar_length=20):
    
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    
    sys.stdout.write("\rPercent: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))  

#Suppresion of irrelavant matplotlib warnings in console
warnings.filterwarnings("ignore", category=mpl.cbook.mplDeprecation)
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.set_printoptions(suppress=True, linewidth = 150)

#Strings to be set dependant on data usage

ddplan = "\\utils\\dd_plan.txt" #Dedispersion Plan Filepath
modelName = "\\utils\\binary_dropout_40_3_conv_64_nodes_1_dense_128_nodes_1557142941"#CNN normalisation(original) saved model Filepath
#modelName = "\\utils\\binary_dropout_40_3_conv_64_nodes_1_dense_128_nodes_1557235933"#CNN normalisation per file

#Loading the CNN classifier model
classifier = load_model(os.getcwd() + modelName)

#Forming array of the data filepaths
source_paths = []   # Array of file paths to be reviewed

# Loops through all .dat files to store them in the 'source_paths' array
for file in glob.glob(os.getcwd() + '\\idir\\' + "*.dat"):
    source_paths.append(file)

timer1 = []
timer2 = []
timer3 = []
timer4 = []
counter=0 #testing

# Loops through the whole file space defined by 'source_paths'
for i in range(len(source_paths)): 
    
    start1 = timer()
    progressBar(i,len(source_paths))
    path_index = i      # Current path index in the source_paths array
    file = source_paths[path_index]     # Setting which file to open
    
    clusterData = cluster(file)
    
    newArr = clusterData[0]
    labels = clusterData[1]
    end1 = timer()
    
    """
    test = newArr[newArr[:,5] == 2]
    
    print(labels)
    print(newArr)
    print(test)"""
    
    timer1.append(end1 - start1)
    
    start2 = timer()
    timer4.append(len(labels))
    for q in range(1,len(labels)):
        start3 = timer()
        label = labels[q]
        cand = newArr[newArr[:,-1] == labels[q]]
        
        #formatting the candidate data so parsable by CNN
        matrix = matrix_form(cand,ddplan)
        matrix = matrix.reshape(-1,100,100,1)/np.amax(matrix)
        
        #predicting with the CNN
        prediction = classifier.predict_classes(matrix)
        
        #prediction gives 0 or 1 corresponding to the index in this array
        class_names = ["RFI","Signal"]
        
        
        """result = writer(label, newArr, 'predict')"""#OSKAR superfluous?
        
        #plot loop if condition is met
        if prediction[0][0] == 1:
            print("pred for" , q ,class_names[prediction[0][0]])
            #print(cand)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(cand[:,0], cand[:,2])
            plt.show
            
        end3 = timer()
        timer3.append(end3 - start3)
        
        counter+=1 #testing 
    



    end2 = timer()
    timer2.append(end2 - start2)

progressBar(1,1)
print(counter)

print("\n")
print("Clustering: ", np.mean(timer1))
print("Loop batch: ", np.mean(timer2))
print("Single loop: ", np.mean(timer3))
print("Len labels: ", np.mean(timer4))



         