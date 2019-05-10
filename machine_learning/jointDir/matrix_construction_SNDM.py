#Matix construction for CNN
#with SN and Width data
import numpy as np
import pandas as pd
import glob, os
import sys
from timeit import default_timer as timer


np.set_printoptions(linewidth = 100)

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

def DF(path):
    """Opens binary encoded file, reshapes it into columns and returns a pandas datafram.

    Keyword arguments:
    path -- the path to the file to be opened
    """
    axislabels = ["DM", "Time", "S/N", "Width","Label"]     # Labels of data types contained in file
    
    Tfile = open(path,'r')  
    data = np.fromfile(Tfile,np.float32,-1)         # Decodes data to float32 objects
    c = data.reshape((-1,5))                        # Reshapes the string of numbers into columns
    df = pd.DataFrame(c,columns=axislabels)         # Creates dataframe
    Tfile.close()
    
    return df

def indexing(arr1, arr2):
    
    index = np.argsort(arr1)
    sorted_arr1 = arr1[index]
    sorted_index = np.searchsorted(sorted_arr1, arr2)
    
    yindex = np.take(index, sorted_index, mode="clip")
    mask = arr1[yindex] != arr2
    
    result = np.ma.array(yindex, mask=mask)
    return result

def import_prompt():
    import_name = input("Name of folder to import from: ")
    while import_name == "":
        import_name = input("Please enter the name of the folder to import from: ")
    exists = os.path.exists(os.getcwd() + '\\sim_data\\' + import_name)
    while exists == False:
        print("Folder does not exist, try again.")
        import_name = input("Name of folder to import from: ")
        exists = os.path.exists(os.getcwd() + '\\sim_data\\' + import_name)
    import_name = os.getcwd() + '\\sim_data\\' + import_name
    
    return import_name
        
def output_prompt():
    output_folder = input("Name of folder to be created: ")
    while output_folder == "":
        output_folder = input("Please enter a name for the folder to be created: ")
    while True:
        try:    # Only creates folders if they don't already exist
            os.mkdir(os.getcwd() + '\\matrix_files\\' + output_folder)
            break
        except:
            print("A folder with this name already exists. Please enter a different name.")
            output_folder = input("Name of folder to be created: ")
     
    return output_folder

#Import data
#array of file locations and chosing the file to inspect with path
source_paths = []
#os.path.exists(path_in_here) 

input_folder = import_prompt()
#filling source_paths from the idir
for file in glob.glob(input_folder + "\\*\\*.dat"):
    source_paths.append(file)


output_folder = output_prompt()
    
# Make directory for matrix files to go to
os.mkdir(os.getcwd()+"\\matrix_files\\" + output_folder + "\\Burst\\")
os.mkdir(os.getcwd()+"\\matrix_files\\" + output_folder + "\\Noise\\")


#import dedispersion plan
df_ddp = pd.read_csv("dd_plan.txt")

#Constructing DF of possible DM values from ddp
DM_start = df_ddp["DM_start"]
DM_stop = df_ddp["DM_stop"]
DM_step = df_ddp["DM_step"]

DM_poss = [0.0] 

for i in range(len(DM_stop)):
    DM_range = DM_stop[i]- DM_start[i]
    num = round(DM_range / DM_step[i])
       
    for j in range(int(num)):
        DM_poss.append(round(DM_poss[-1] + DM_step[i],3))

possDF = pd.DataFrame(DM_poss,columns = ["DM"])



timer_1 = [] #creating the first zero
timer_2 = [] #time downsample/pad
timer_3 = [] #dm downsample/pad
timer_4 = [] #save
timePerK = []
pad7 = []

#y = 20000
n_s = [] ###testing dm dimensions
for x in range(len(source_paths)):    
#for x in range(y):
    #print(x)
    #reading in data files    

    progressBar(x,len(source_paths))
    #progressBar(x, y)
    start5 = timer()
    
    
    clusterDF = DF(source_paths[x])
    """print(clusterDF)"""
    clusterDF = clusterDF.sort_values(by = "DM")
    clusterDF = clusterDF.reset_index(drop = True)
    clusterDF["DM"] = clusterDF["DM"].astype(float).round(3) #have to round fp error out of dm
    #set up arrays from data
    DM = np.array(clusterDF["DM"], dtype = np.float64)
    TIME = np.array(clusterDF["Time"], dtype = np.float64)
    SN = np.array(clusterDF["S/N"], dtype = np.float64).round(0)
    WIDTH = np.array(clusterDF["Width"], dtype = np.float64).round(2)
    
    
    dmRange = np.array([DM[0],DM[-1]])

    # Getting dm range, n dimension value
    botDMind = possDF.loc[possDF["DM"] == dmRange[0]].index.item()
    topDMind = possDF.loc[possDF["DM"] == dmRange[1]].index.item()
    rows = topDMind - botDMind

    #print(np.unique(SN))
    #print(WIDTH)
    SN = SN.astype(int)
    #rows = WIDTH.max()
    columns = SN.max()
    
    zero = np.zeros((rows+1,columns+1))
    
    testDM = np.array(possDF["DM"].values)
    indices = indexing(testDM,DM) - botDMind
    dmArange = np.arange(len(DM))
    
    r_SNpixels = 100
    r_DMpixels = 100
    
    zero[indices,SN[:]] = WIDTH[dmArange] 
    
    #case of Time padding
    if columns < r_SNpixels:
        zero2 = np.zeros((rows + 1,r_SNpixels))
        pad = (r_SNpixels - columns )/2
        pad = round(pad) #amount to pad from the top (~equal on the bottom)
        for q in range(len(zero[0,:])):
            ##iterator over num of columns
            zero2[:,q] = zero[:,q]
    #case of Time downsample 
    else:
        zero2 = np.zeros((r_SNpixels, rows + 1))
        timeSplit = np.array_split(zero[:,:], r_SNpixels, axis = 1)
        for k in range(len(timeSplit)):
            zero2[k] = (timeSplit[k].max(1))
        
        zero2 = np.transpose(zero2)
 
     #case of DM padding
    if rows < r_DMpixels:
        zero3 = np.zeros((r_DMpixels,r_SNpixels))
        pad = (r_DMpixels-rows)/2
        pad = round(pad) #amount to pad from the top (~equal on the bottom)
        for q in range(len(zero2[:,:])):
            ##iterator over num of rows
            zero3[q+pad,:] = zero2[q,:]
    #case of DM downsample      
    else:
        dmSplit = np.array_split(zero2[:,:], r_DMpixels, axis = 0)
        zero3 = np.zeros((r_DMpixels,r_SNpixels))
        for k in range(len(dmSplit)):
            zero3[k] = (dmSplit[k].max(0))
    
    c_id = clusterDF["Label"].values
    if c_id.any() == 1:
        c_id = str(1) #Case of label array containing a 1 i.e a generated burst
        
    else:
        c_id = str(0) #Case of label array containing only 0 i.e generated noise
    
    n_s.append(zero3.shape)#(len(zero3[:,:]),len(zero3[0,:]))) ###testing
    new_name = source_paths[x].split("\\")[-1].replace(".dat", "_m_" + c_id )
    
    #temporary for playground
    if c_id == "0":
        np.save(os.getcwd()+"\\matrix_files\\" + output_folder + "\\Noise\\" + new_name, zero3)
    else:
        np.save(os.getcwd()+"\\matrix_files\\" + output_folder + "\\Burst\\"+ new_name, zero3)
    
   
     
print("\n")  
print(np.unique(n_s)) ###testing