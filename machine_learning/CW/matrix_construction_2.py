#Matix construction for CNN
import numpy as np
import pandas as pd
import glob, os
import sys
from timeit import default_timer as timer


np.set_printoptions(linewidth = 100)

###NEED TO HANDLE SEPERATE CLUSTERS SO ADD COLUMN TO DATA FILES WITH LABEL
###see sourcepaths[16]&8
def progressBar(value, endvalue, bar_length=20):
    
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    
    sys.stdout.write("\rPercent: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))


# Creates dataframe for file
def DF(path):
    axislabels = ["DM", "Time", "S/N", "Width","Label"]
    Tfile = open(path,'r')
    data = np.fromfile(Tfile,np.float32,-1) 
    c = data.reshape((-1,5))
    df = pd.DataFrame(c,columns=axislabels)
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

#Import data
#array of file locations and chosing the file to inspect with path
source_paths = []
Folder = "training_2"
"""
#file number in source_paths to open
x = 0
"""

#filling source_paths from the idir
for file in glob.glob(os.getcwd() + f"\{Folder}\\"+ "*.dat"):
    
    source_paths.append(file)
    
#make directory for matrix files to go to
try:
    os.mkdir(os.getcwd()+"\matrix_files\\Playground2\\")
    os.mkdir(os.getcwd()+"\matrix_files\\Playground2\\Burst\\")
    os.mkdir(os.getcwd()+"\matrix_files\\Playground2\\Noise\\")
   
except:
    pass

#import dedispersion plan
df_ddp = pd.read_csv("dd_plan.txt")
"""print(df_ddp)"""
#setup array for step limits
dd_DM = np.array(df_ddp["DM_stop"])
dd_step = np.array(df_ddp["DM_step"])


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
        
"""print(DM_poss)"""
 
possDF = pd.DataFrame(DM_poss,columns = ["DM"])
"""print(DF.head)"""


timer_1 = [] #creating the first zero
timer_2 = [] #time downsample/pad
timer_3 = [] #dm downsample/pad
timer_4 = [] #save




y = 40
n_s = [] ###testing dm dimensions
for x in range(0,y):    
    #reading in data files    
    #progressBar(x,y)
    
    
    clusterDF = DF(source_paths[x])
    """print(clusterDF)"""
    clusterDF = clusterDF.sort_values(by = "DM")
    clusterDF = clusterDF.reset_index(drop = True)
    clusterDF.DM = clusterDF.DM.astype(float).round(3) #have to round fp error out of dm
    #set up arrays from data
    DM = np.array(clusterDF["DM"])
    TIME = np.array(clusterDF["Time"])
    SN = np.array(clusterDF["S/N"])
    
    
    #find ranges of time and dm
    #already sorted by dm so use difference between first and last row
    """
    df_dmrange = pd.concat([df.head(1),df.tail(1)])
    dmrange = np.array(df_dmrange["DM"])
    print(dmrange)
    """
    dmRange = np.array([DM[0],DM[-1]])

    # Getting dm range, n dimension value
    botDMind = possDF.loc[possDF["DM"] == dmRange[0]].index.item()
    topDMind = possDF.loc[possDF["DM"] == dmRange[1]].index.item()
    rows = topDMind - botDMind

    # Time range, m dimension 
    ###same as before
    t_step = 256*(pow(10,-6))
    
    botTimeInd = round(np.amin(TIME)/t_step)
    topTimeInd = round(np.amax(TIME)/t_step)
   
    columns = topTimeInd - botTimeInd
    columns = int(columns)

    
    # Matrix construction
    # Zero matrix of required dimension
    ###+1 to handle the fact you need to count the first value istelf
    
    zero = np.zeros((rows + 1,columns + 1))
    zeroAlt = np.zeros((rows + 1,columns + 1))
    
    # Find position of data point
    
    
    #h = DM_poss[DM_poss == DM[:]]
    v = np.round(TIME/t_step) 
    v = v[:] - botTimeInd
    #print(v)
    
    
    
    start_1=timer()
    testDM = np.array(possDF["DM"].values)
    indices = indexing(testDM,DM) - botDMind
    dmArange = np.arange(len(DM))
    zeroAlt[indices,v.astype(int)] = SN[dmArange]
    end_1 = timer()

    """
    for l in range(len(DM)):
        #dm axis location
        u = possDF.loc[possDF["DM"] == DM[l]].index.item() - botDMind
        zero[u][int(v[l])]=SN[l]
        
    print((zero == zeroAlt).all())
    """
    timer_1.append(end_1-start_1)

    
    ###need to downsample in time by some bin amount
    #pixels wanted in time axis
    
    r_Tpixels=100
    
    #new time downsampled matrix of wanted size
    zero2=np.zeros((rows + 1,r_Tpixels))
    
    #case of needing to pad
    if columns < r_Tpixels:
        start_2 = timer()
        pad = (r_Tpixels - columns )/2
        pad = round(pad) #amount to pad from the top (~equal on the bottom)
        for q in range(len(zero[0,:])):
            ##iterator over num of columns
            zero2[:,q+pad] = zero[:,q]
    #case of Time downsample 
        end_2 = timer()
        timer_2.append(end_2 - start_2)
    
    else:
        start_4 = timer()
        for p in range(len(zero[:,:])):
            ##iterator over num of rows
            #downsample into r_pixels in time axis
            dummy = np.array_split(zero[p,:],r_Tpixels)
            dummy2 = []
            for j in range(len(dummy)):
                #take max from each pixel group to give value to new pixel
                dummy2.append(max(dummy[j]))
            #fill new matrix by rows
            zero2[p] = dummy2
        end_4 = timer()
        timer_4.append(end_4 - start_4)
    
    ###need to downsample or pad DM dimensions or nothing if n=r_DMpixels
    #pixels wanted in dm axis
    start_3 = timer()
    r_DMpixels = 70
    
    #new matrix of required dimensions
    zero3 = np.zeros((r_DMpixels,r_Tpixels))
    #case of needing to pad
    
    if rows < r_DMpixels:
        pad = (r_DMpixels-rows)/2
        pad = round(pad) #amount to pad from the top (~equal on the bottom)
        for q in range(len(zero2[:,:])):
            ##iterator over num of rows
            zero3[q+pad,:] = zero2[q,:]
            
    #case of DM downsample      
    else:
        for i in range(len(zero2[0,:])):
            ##iterator over num of columns
            #downsample into r_pixels in time axis
            dummy = np.array_split(zero2[:,i],r_DMpixels)
            dummy2 = []
            for j in range(len(dummy)):
                dummy2.append(max(dummy[j]))
            zero3[:,i] = dummy2
    end_3 = timer()
    timer_3.append(end_3 - start_3)
    #name of matrix file
    
    c_id = clusterDF["Label"].values
    if c_id.any() == 1:
        c_id = str(1) #Case of label array containing a 1 i.e a generated burst
        
    else:
        c_id = str(0) #Case of label array containing only 0 i.e generated noise
        
    n_s.append((len(zero3[:,:]),len(zero3[0,:]))) ###testing
    new_name = source_paths[x].split(f"{Folder}\\")[1].replace(".dat","_m_"+c_id)
    #np.savetxt(os.getcwd()+"\matrix_files\\Final\\"+new_name, zero3)

    """
    #temporary for playground
    if c_id == "0":
        np.save(os.getcwd()+"\matrix_files\\Playground2\\Noise\\"+new_name, zero3)
    else:
        np.save(os.getcwd()+"\matrix_files\\Playground2\\Burst\\"+new_name, zero3)"""
    
print(np.unique(n_s)) ###testing

print(f"Timer 1, read in and form first matrix: {np.mean(timer_1)} s" )
print(f"Timer 2, Time DS and pad: {np.mean(timer_2)} s" )
print(f"Timer 3, DM DS and pad: {np.mean(timer_3)} s" )
print(f"Timer 4, write files: {np.mean(timer_4)} s" )
#print(h)