# Prediction formatting
# A script to fromat clustered signal candidates into the required matrix
# format and shape for the CNN to perform the prediction of type

###imports
import numpy as np
import pandas as pd
import os

def indexing(arr1, arr2):
    #returns index's of matching array values
    index = np.argsort(arr1)
    sorted_arr1 = arr1[index]
    sorted_index = np.searchsorted(sorted_arr1, arr2)
    
    yindex = np.take(index, sorted_index, mode="clip")
    mask = arr1[yindex] != arr2
    
    result = np.ma.array(yindex, mask=mask)
    return result

def de_des_plan(ddFile):
    #Forms array of possible DM values for a specific dedispersion plan
    
    #import dedispersion plan
    df_ddp = pd.read_csv(os.getcwd() + ddFile)

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
    
    return possDF

def matrix_form(ClusterArr, ddFile):
    #Forms a Matrix given a candidate array and filename of the dedispersion plan in the directory
    #Matrix Dimensions of final matrix in Time(columns) and DM(rows)
 
    #Forming data arrays form the candidate
    
    SN = ClusterArr[:,2].round(0)
    WIDTH = ClusterArr[:,3].round(0)
    
    SN = SN.astype(int)
    WIDTH = WIDTH.astype(int)
    rows = WIDTH.max()
    columns = SN.max()
    
    zero = np.zeros((rows+1,columns+1))
        
    r_SNpixels = 100
    r_Wpixels = 32
    
    zero[WIDTH[:],SN[:]] = 1 
    
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
    if rows < r_Wpixels:
        zero3 = np.zeros((r_Wpixels,r_SNpixels))
        pad = (r_Wpixels-rows)/2
        pad = round(pad)
        pad = int(pad)#amount to pad from the top (~equal on the bottom)
        for q in range(len(zero2[:,:])):
            ##iterator over num of rows
            zero3[q+pad,:] = zero2[q,:]
    #case of DM downsample      
    else:
        dmSplit = np.array_split(zero2[:,:], r_Wpixels, axis = 0)
        zero3 = np.zeros((r_Wpixels,r_SNpixels))
        for k in range(len(dmSplit)):
            zero3[k] = (dmSplit[k].max(0))
    
    clusterMatrix = zero3 #simple rename
    
    return clusterMatrix




