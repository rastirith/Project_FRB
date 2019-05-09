import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import glob, os, sys
import warnings
import pickle

from clustering import cluster
from matrix import matrix_form
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
#modelName = "\\utils\\binary_dropout_40_3_conv_64_nodes_1_dense_128_nodes_1557322926" #full z-score (withzero)
#modelName = "\\utils\\binary_dropout_40_3_conv_64_nodes_1_dense_128_nodes_1557343773" # new noise with norm
modelName = "\\utils\\binary_dropout_40_3_conv_64_nodes_1_dense_128_nodes_1557361636"

#Loading the CNN classifier model
classifier = load_model(os.getcwd() + modelName)

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

# Loops through the whole file space defined by 'source_paths'
for i in range(5): 

    #progressBar(i,len(source_paths))
    path_index = i      # Current path index in the source_paths array
    file = source_paths[path_index]     # Setting which file to open
    
    clusterData = cluster(file)
    
    newArr = clusterData[0]
    labels = clusterData[1]

    for q in range(1,len(labels)):
        label = labels[q]
        cand = newArr[newArr[:,-1] == label]

        #formatting the candidate data so parsable by CNN
        matrix = matrix_form(cand,ddplan)
        matrix = matrix.reshape(-1,100,100,1) #/0.71 #this mean value
        matrix /= 17.94
        #print(matrix[np.nonzero(matrix)])
        #break
        
        #matrix = (matrix - scaling[0])/scaling[1]
        
        
        #predicting with the CNN
        prediction = classifier.predict_classes(matrix)
        #print(prediction)
        #prediction gives 0 or 1 corresponding to the index in this array
        class_names = ["RFI","Signal"]
        
        #plot loop if condition is met
        if prediction[0][0] == 1:
            print("pred for" , q ,class_names[prediction[0][0]])
            #print(cand)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(cand[:,0], cand[:,2])
            plt.show
            
        
        counter+=1 #testing 
    



progressBar(1,1)
print(counter)


         