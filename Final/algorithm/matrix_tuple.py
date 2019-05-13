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
    r_Tpixels = 100
    r_DMpixels = 100
    
    #Forming data arrays form the candidate
    DM = ClusterArr[:,0].round(3)
    TIME = ClusterArr[:,1]
    SN = ClusterArr[:,2]
    WIDTH = ClusterArr[:,3].round(1)
    #DM range spanned by cluster
    dmRange = np.array([DM[0],DM[-1]])
   
    #Array of all possible DM values from the dedespersion plan
    possDF = de_des_plan(ddFile)
    
    
    #Finding the numeber rows for first candidate matrix from DM range of cand
    botDMind = possDF.loc[possDF["DM"] == dmRange[0]].index.item()
    topDMind = possDF.loc[possDF["DM"] == dmRange[1]].index.item()
    rows = topDMind - botDMind
    
    #Value of time resolution in data
    t_step = 256*(pow(10,-6))
    
    #Finding the number of columns for first candidate matrix from candidate time span
    botTimeInd = (np.amin(TIME)/t_step)
    topTimeInd = (np.amax(TIME)/t_step)
    columns = int(round(topTimeInd - botTimeInd))
    
    #matrix to be filled by candidate data
    zero = np.zeros((rows + 1,columns + 1,2))
    
    #index of each data point in time space
    v = np.round(TIME/t_step)
    v = v[:] - botTimeInd
    
    
    testDM = np.array(possDF["DM"].values)
    indices = indexing(testDM,DM) - botDMind #getting index of data points in DM space
    dmArange = np.arange(len(DM))
    
    tupleArr = np.zeros((len(DM),2))
    tupleArr[:,0] = SN[dmArange]
    tupleArr[:,1] = WIDTH[dmArange]
    
    zero[indices,v.astype(int)] = tupleArr[dmArange] #filling zero array with cand data
    
    #Shaping the array to the required final dimensions
    #first along Time axis then along DM axis
   
    #case of Time padding
    if columns < r_Tpixels:
        zero2=np.zeros((rows + 1,r_Tpixels,2))
        pad = (r_Tpixels - columns )/2
        pad = round(pad) #amount to pad from the top (~equal on the bottom)
        for q in range(len(zero[0,:])):
            ##iterator over num of columns
            zero2[:,q+pad] = zero[:,q]
    #case of Time downsample 
    else:
        zero2 = np.zeros((r_Tpixels, rows + 1,2))
        timeSplit = np.array_split(zero[:,:], r_Tpixels, axis = 1)
        for k in range(len(timeSplit)):
            zero2[k] = (timeSplit[k].max(1))
        
        zero2 = np.transpose(zero2, (1,0,2))
 
    #case of DM padding
    if rows < r_DMpixels:
        zero3 = np.zeros((r_DMpixels,r_Tpixels,2))
        pad = (r_DMpixels-rows)/2
        pad = round(pad) #amount to pad from the top (~equal on the bottom)
        for q in range(len(zero2[:,:])):
            ##iterator over num of rows
            zero3[q+pad,:] = zero2[q,:]
    #case of DM downsample      
    else:
        dmSplit = np.array_split(zero2[:,:], r_DMpixels, axis = 0)
        zero3 = np.zeros((r_DMpixels,r_Tpixels,2))
        for k in range(len(dmSplit)):
            zero3[k] = (dmSplit[k].max(0))
    
    clusterMatrix = zero3 #simple rename
    
    return clusterMatrix




