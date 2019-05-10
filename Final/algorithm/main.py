import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import glob, os, sys
import warnings
import pickle

from clustering import cluster
from matrix_SN_DM_dimension import matrix_form
#from matrix import matrix_form
from matrix_widthSN import matrix_form as MF2
from tensorflow.keras.models import load_model

from timeit import default_timer as timer

def progressBar(value, endvalue, bar_length=20):
    """Displays and updates a progress bar in the console window.

    Keyword arguments:
    value -- current iteration value
    endvalue -- value at which the iterations end
    bar_length -- length of progress bar in console window
    """
    
    percent = float(value) / endvalue       # Calculates progress percentage
    arrow = '-' * int(round(percent * bar_length) - 1) + '>'    # Draws arrow displayed
    spaces = ' ' * (bar_length - len(arrow))
    
    # Writes/updates progress bar
    sys.stdout.write("\rPercent: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))

#Suppresion of irrelavant matplotlib warnings in console
warnings.filterwarnings("ignore", category=mpl.cbook.mplDeprecation)
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.set_printoptions(suppress=True, linewidth = 150)

#Strings to be set dependant on data usage
ddplan = "\\utils\\dd_plan.txt" #Dedispersion Plan Filepath
#modelName = "\\utils\\binary_dropout_40_3_conv_64_nodes_1_dense_128_nodes_1557142941"#CNN normalisation(original) saved model Filepath
#modelName = "\\utils\\binary_dropout_40_3_conv_64_nodes_1_dense_128_nodes_1557235933"#CNN normalisation per file
#modelName = "\\utils\\binary_dropout_40_3_conv_64_nodes_1_dense_128_nodes_1557316602" #CNN Z-score norm
#modelName = "\\utils\\binary_dropout_40_3_conv_64_nodes_1_dense_128_nodes_1557317452" #osk meandiv CNN
#modelName = "\\utils\\binary_dropout_40_3_conv_64_nodes_1_dense_128_nodes_1557322428" #no norm
#modelName2 = "\\utils\\binary_dropout_40_3_conv_64_nodes_1_dense_128_nodes_1557322926" #full z-score (withzero)
#modelName = "\\utils\\binary_dropout_40_3_conv_64_nodes_1_dense_128_nodes_1557343773" # new noise with norm

#modelName = "\\utils\\binary_dropout_40_3_conv_64_nodes_1_dense_128_nodes_1557405243" #with width


modelName1 = "\\utils\\binary_dropout_40_3_conv_64_nodes_1_dense_128_nodes_1557413490" #new width vs SN
#modelName2 = "\\utils\\binary_dropout_40_3_conv_64_nodes_1_dense_128_nodes_1557448728" # snDM avg (~25)

modelName2 = "\\utils\\binary_dropout_40_3_conv_64_nodes_1_dense_128_nodes_1557451692"

#Loading the CNN classifier model
classifier1 = load_model(os.getcwd() + modelName1)
classifier2 = load_model(os.getcwd() + modelName2)

step1Count = 0
predCount = 0

#Scaling parameters for the data to be parsed by the CNN
pickle_in = open("utils\\scale_param.pickle","rb")
scaling = pickle.load(pickle_in)
pickle_in.close()

#Forming array of the data filepaths
source_paths = []   # Array of file paths to be reviewed

# Loops through all .dat files to store them in the 'source_paths' array
for file in glob.glob(os.getcwd() + '\\idir\\' + "*.dat"):
    source_paths.append(file)

counter=0 #testing

correct=0
incorrect=0
RFIcount=0
step1Miss=0
step2Miss=0
signal1=0
signal2 = 0

# Loops through the whole file space defined by 'source_paths'

for i in range(len(source_paths)): 
    
    progressBar(i,len(source_paths))

    path_index = i      # Current path index in the source_paths array
    file = source_paths[path_index]     # Setting which file to open
    
    clusterData = cluster(file)
    
    newArr = clusterData[0]
    labels = clusterData[1]
    
    for q in range(1,len(labels)):
        label = labels[q]
        cand = newArr[newArr[:,-1] == label]

        if 1 in cand[:,4]:
            signal2 +=1
        """
        #formatting the candidate data so parsable by CNN
        matrix = MF2(cand,ddplan)

        #matrix = matrix.reshape(-1,100,100,2) #/0.71 #this mean value
        matrix = matrix.reshape(-1,32,100,1)
        
        #matrix = (matrix - scaling[0])/scaling[1]
        
        #predicting with the CNN
        prediction = classifier1.predict_classes(matrix)
        #print(prediction)
        #prediction gives 0 or 1 corresponding to the index in this array
        class_names = ["RFI","Signal"]
        
        #plot loop if condition is met
        if prediction[0][0] == 1:# | prediction[0][0] ==  1:"""
        step1Count += 1
        
        #formatting the candidate data so parsable by CNN
        matrix = matrix_form(cand,ddplan)

        matrix = matrix.reshape(-1,100,100,1) #/0.71 #this mean value
        
        
        #matrix = (matrix - scaling[0])/scaling[1]
        matrix = matrix/25.93
        
        #predicting with the CNN
        prediction = classifier2.predict_classes(matrix)
        #print(prediction)
        #prediction gives 0 or 1 corresponding to the index in this array
        class_names = ["RFI","Signal"]
        
        if prediction[0][0] == 1:
            predCount += 1
            if 1 in cand[:,4]:
                correct +=1
                """
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.scatter(cand[:,0], cand[:,2])
                
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.scatter(newArr[:,1], newArr[:,0], s = 4, c = "r", alpha = 0.6)
                ax.scatter(cand[:,1], cand[:,0], s = 4, c = "g", alpha = 0.6)
                plt.show"""
            else:
                """
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.scatter(cand[:,0], cand[:,2])
                
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.scatter(newArr[:,1], newArr[:,0], s = 4, c = "r", alpha = 0.6)
                ax.scatter(cand[:,1], cand[:,0], s = 4, c = "g", alpha = 0.6)
                plt.show"""
                incorrect +=1
            #print("pred for" , q ,class_names[prediction[0][0]])
            #print(cand)
            """
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(cand[:,0], cand[:,2])
            plt.show"""
            #break
        elif 1 in cand[:,4]:
            """
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(cand[:,0], cand[:,2])
            
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(newArr[:,1], newArr[:,0], s = 4, c = "r", alpha = 0.6)
            ax.scatter(cand[:,1], cand[:,0], s = 4, c = "g", alpha = 0.6)
            plt.show"""
            step2Miss +=1    
        else:
            RFIcount+=1
        """
        elif 1 in cand[:,4]:
            step1Miss +=1
        else:
            RFIcount+=1"""
        counter+=1 #testing 
    #break



progressBar(1,1)
print("\n clusters: ",counter,
      "\n group w\1: ",signal2,
      "\n step1count: ",step1Count,
      "\n pred counter: ",predCount,
      "\n RFI count: ",RFIcount,
      "\n correct: ",correct,
      "\n incorrect: ",incorrect,
      "\n step1Miss: ",step1Miss,
      "\n step2Miss: ",step2Miss,)


         