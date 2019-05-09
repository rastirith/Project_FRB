#Formating Training Data
#reshape the constructed matrix files for use in CNN 
#saving with pickle
import numpy as np
import os
import random
import pickle
import sys

def progressBar(value, endvalue, bar_length=20):
    
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    
    sys.stdout.write("\rPercent: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))


DATADIR = os.getcwd() + "\matrix_files\\val_test\\"

#arrLength = 5

CATEGORIES=["Noise","Burst"]

def import_prompt():
    import_name = input("Name of folder to import from: ")
    while import_name == "":
        import_name = input("Please enter the name of the folder to import from: ")
    exists = os.path.exists(os.getcwd() + '\\matrix_files\\' + import_name)
    while exists == False:
        print("Folder does not exist, try again.")
        import_name = input("Name of folder to import from: ")
        exists = os.path.exists(os.getcwd() + '\\matrix_files\\' + import_name)
    
    return import_name
        
def output_prompt():
    output_folder = input("Name of folder to be created: ")
    while output_folder == "":
        output_folder = input("Please enter a name for the folder to be created: ")
    while True:
        try:    # Only creates folders if they don't already exist
            os.mkdir(os.getcwd() + '\\formed_data\\' + output_folder)
            break
        except:
            print("A folder with this name already exists. Please enter a different name.")
            output_folder = input("Name of folder to be created: ")
            
        return output_folder
    
def create_training_data(DATADIR):
    matrix_data = []
    label_data = []
    for category in CATEGORIES:
        count = 0
        path = os.path.join(DATADIR,category)
        class_num = CATEGORIES.index(category)
        for matrix in os.listdir(path):
            break
            count += 1
            progressBar(1,len(os.listdir(path)))
            #if len(matrix_data)==2:
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

input_folder = import_prompt()
output_folder = output_prompt()

data = create_training_data(input_folder)   

X = data[0]
y = data[1]

X = X.reshape(-1, 100, 100, 1)

#download more ram or batch it 
X = X/np.mean(X)
#print(np.unique(X[0]))

pickle_out = open("formed_data\\" + output_folder + "\\X_scaled.pickle","wb")
pickle.dump(X,pickle_out, protocol=4)
pickle_out.close()

pickle_out = open("formed_data\\" + output_folder + "\\y2.pickle","wb")
pickle.dump(y, pickle_out, protocol=4)
pickle_out.close()


