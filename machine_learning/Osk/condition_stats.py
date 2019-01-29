#Cluster statistics
#Running DBscan and conditions over the test files to gain statistics of how 
#the number of clusters are reduced by each step
#Also assessing the S/N point scores frequencies 

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import glob, os
from sklearn import preprocessing 

# Returns 'num' lowest elements of array 'arr' in a new array
def sort(arr,num):
    xsorted = np.sort(arr)[:num]
    return xsorted

def clusterOrder(clusterArr):
    
    lab_arr = np.unique(clusterArr)
    
    for i in range(len(clusterArr)): 
        for q in range(1,len(lab_arr)):
            if (lab_arr[q] == clusterArr[i]):
                clusterArr[i] = q - 1
                break

def clusterSort(clusterArr, pointsArr):
    temp_arr = []     # Array of arrays containing points for each label. e.g. labels_arr[0] contains all points with label -1
    lab_arr = np.unique(clusterArr)

    for i in range(len(np.unique(clusterArr))):
        temp_list = []
        temp_arr.append(temp_list)
    
    for i in range(len(clusterArr)):
        for q in range(len(lab_arr)):
            if (clusterArr[i] == lab_arr[q]):
                temp_arr[q].append(pointsArr[i])
                break

    for q in range(len(temp_arr)):
        temp_arr[q] = np.array(temp_arr[q])
    
    temp_arr = np.array(temp_arr)
    return temp_arr

#creates dataframe for file
def DF(path):
    axislabels = ["DM", "Time", "S/N", "Width"]
    Tfile = open(path,'r')
    data = np.fromfile(Tfile,np.float32,-1) 
    c = data.reshape((-1,4))
    df = pd.DataFrame(c,columns=axislabels)
    Tfile.close()
    return df

def ROWFORM(cluster_list,column_variable,array):
    data = len(cluster_list)
    array.append(data)

#array of file locations and chosing the file to inspect with path
source_paths = []

#DataFrame utility
df2_labels=['File_Path','DBSCAN','Noise','DMlimit','Peak location','Shape & Sharpness','NRFIclusters','REJECTED','Fair','Good','Excellent','bin1','bin2','bin3','bin4','bin5','bin6','bin7','bin8','bin9','bin10']
df2 = pd.DataFrame(columns=df2_labels)
#df2_data = [] #store a row of data per file to append to DF2 
#print(df2.head())

#filling source_paths from the idir
for file in glob.glob(os.getcwd() + '\idir\\' + "*.dat"):
    source_paths.append(file)

#ratios = []
   
for i in range(0,72): 
    #Variables for counting
    #step 1
    DBclusters = 0
    #Sum_DBclusters = 0
    #step 2
    Noiseclusters = 0
    #Sum_Noiseclusters = 0
    #step 3
    DMlimitclusters = 0
    #Sum_DMlimitclusters = 0
    #step 4
    SNPclusters = 0
    #Sum_SNPclusters = 0
    #step 5
    REJECTED = 0
    FAIRclusters = 0
    GOODclusters = 0
    EXCELclusters = 0
    SNRclusters = 0
    #step 6
    NRFIclusters = 0
    #step 7
    Pbin= 10*[0] #array to store a count of singals by confidence percentiles
    
    #Store a row of data per file to append to DF2    
    df2_data = []  
    
    #path to file variable
    path=i
    print(i)
    df2_data.append(i)
    #setting which file to open
    FILE = source_paths[path]
    #getting df for test file
    df = DF(FILE)
    #getting arrays from df
    X_db = np.array(df.drop(columns=['Width', 'S/N']))
    X = np.array(df.drop(columns=['Width']))
    
    #sorted by DM
    points_db = X_db[X_db[:,0].argsort()]
    points = X[X[:,0].argsort()]
    
    #memory handling
    X_db =[]
    X = []
    del df
    
    dm_lim = 0.03*max(points_db[:,0])
    points_new = []
    #print("Dm limit 1: " + str(round(dm_lim,1)))
    
    for i in range(len(points_db)):
        if (points_db[i][0] > dm_lim):
            points_new.append(points_db[i])
           
    points_new = np.array(points_new)
    
    X_scaled = preprocessing.MinMaxScaler().fit_transform(points_new) #Rescales the data so that the x- and y-axes get ratio 1:1
    
    xeps = 0.025    # Radius of circle to look around for additional core points
    xmin = 2        # Number of points within xeps for the point to count as core point
    
    clusters = DBSCAN(eps=xeps, min_samples = xmin, n_jobs = -1).fit_predict(X_scaled)   
    #plt.scatter(X_scaled[:, 1], X_scaled[:, 0], c=clusters, cmap="Paired", alpha = 0.4, vmin = -1, s = 15)
    
    #more memory
    points_new = []
    X_scaled = []
    
    # Re-inserts bottom points with labels -1 for RFI
    length = len(points) - len(clusters)
    clusters = np.insert(clusters,0,np.full(length,-1))
    
    labels_arr = clusterSort(clusters, points)
    clusterOrder(clusters)
    
    #DBclusters
    ROWFORM(labels_arr,DBclusters,df2_data)
    #DBclusters = len(labels_arr)
    #df2_data.append(DBclusters)
    
    #Noise condition for Nevents<Nmin =>noise
    N_min = 20 #tuneable number (set to match Karako)
    for q in range(1,len(np.unique(clusters))):
        if (len(labels_arr[q]) < N_min):
            for i in range(len(clusters)):
                if (clusters[i] == q - 1):
                    clusters[i] = -1
    #Re-order        
    labels_arr = clusterSort(clusters, points)
    clusterOrder(clusters)
    
    #Noiseclusters
    ROWFORM(labels_arr,Noiseclusters,df2_data)
              
    dm_lim = 40         # Increased DM-limit to check for clusters with 'fraction' of their points below this limit
    fraction = 0.05 
    
    # Condition that sets all clusters with 'fraction' of its points below dm_lim to also be classified as RFI
    for q in range(1,len(np.unique(clusters))):
        num_temp = int(round(fraction*len(labels_arr[q]),0))    # Calculates how many points of a label a certain fraction corresponds to and rounds to nearest integer
        temp = sort(labels_arr[q][:,0],num_temp)                # Returns the 'num_temp' lowest dms in labels_arr[q]
    
        if ((len(temp) > 0) and (max(temp) < dm_lim)):          # If the highest number in temp is below dm_lim then so is the rest in 'temp'
    
            for i in range(len(clusters)):
                if (clusters[i] == q - 1):
                    clusters[i] = -1
                    
    # Condition that sets all points with dm below dm_lim to be classified as RFI    
    for i in range(len(clusters)):
        if (points[i][0] <= dm_lim):
            clusters[i] = -1
     
    labels_arr = clusterSort(clusters, points)
    clusterOrder(clusters)
    
    #DMlimitclusters
    ROWFORM(labels_arr,DMlimitclusters,df2_data)
    
    #Conditions for SN
    for q in range(1,len(np.unique(clusters))):
    
        signalToDm = list(zip(labels_arr[q][:,0], labels_arr[q][:,2]))
        signalToDm = np.array(signalToDm)
        min_val = min(signalToDm[:,1])
        #sharpness
        scaled_signal = preprocessing.MinMaxScaler().fit_transform(signalToDm)
        
        #y=0 for visualisation
        for i in range(len(signalToDm[:,1])):
            signalToDm[:,1][i] = signalToDm[:,1][i] - min_val
            
        #splitting into chunks
        split_param = 7 #parameter to determine splitting of cluster for analysis
        dummy = np.array_split(signalToDm,split_param) #shape
        dummy2 = np.array_split(scaled_signal,split_param) #sharpness
        #shape
        meanSN = []
        meanDM = []
        #sharpness
        s_meanSN = []
        s_meanDM = []
        for i in range(len(dummy)):
            #shape
            tempSN = np.mean(dummy[i][:,1])
            tempDM = np.mean(dummy[i][:,0])
            meanSN.append(tempSN)
            meanDM.append(tempDM)
            #sharpness
            s_tempSN = np.mean(dummy2[i][:,1])
            s_tempDM = np.mean(dummy2[i][:,0])
            s_meanSN.append(s_tempSN)
            s_meanDM.append(s_tempDM)
        
        max_val = max(meanSN + min_val)
        
        #Condition for location of peak S/N
        max_ind = np.argmax(meanSN)
        if (max_ind > 4) or (max_ind < 2):
            for i in range(len(clusters)):
                if (clusters[i] == q - 1):
                    clusters[i] = -1
        
        #developing peak shape conditions
        else:
            weight_1 = -1
            weight_2 = -0.3
            weight_3 = 1
            weight_4 = -1
            check_1 = 0.075
            check_2 = 0.15
            score = [0,1.3,2.5,2.5]
            max_score = 2*(score[0] + score[1] + score[2])
            rating = 0
            #bins left side of peak
            for i in range(max_ind - 1, -1, -1):
                ratio=meanSN[i]/meanSN[i+1]
            
                if ((ratio>=(1-check_1)) and (ratio<=1)):
                    rating += weight_1*score[max_ind-(i+1)]
                elif ((ratio>=(1-check_2)) and (ratio<=1)):
                    rating += weight_2*score[max_ind-(i+1)]
                elif (ratio<=1):
                    rating += weight_3*score[max_ind-(i+1)]
                else:
                    rating += weight_4*score[max_ind-(i+1)]
            #bins right side of peak
            for i in range((max_ind+1),split_param):
                ratio=meanSN[i]/meanSN[i-1]

                if ((ratio>=(1-check_1)) and (ratio<=1)):
                    rating += weight_1*score[i-max_ind-1]
                elif ((ratio>=(1-check_2)) and (ratio<=1)):
                    rating += weight_2*score[i-max_ind-1]
                elif ratio <=1:
                    rating += weight_3*score[i-max_ind-1]
                else:
                    rating += weight_4*score[i-max_ind-1]
            
            if rating <0:
                rating =0
            
            #sharpness
            diff_SN = max(s_meanSN) - (0.5*s_meanSN[0] + 0.5*s_meanSN[-1])
            diff_DM = s_meanDM[-1] - s_meanDM[0] #?????center this around peak
            sharp_ratio = diff_SN/(diff_SN+diff_DM) #height/width
            #ratios.append(sharp_ratio)    
            
            #confidence value
            shape_conf = rating/max_score
            tot_conf = 0.743*shape_conf + 0.257*sharp_ratio

            #confidence conditions
            conf_lim0 = 0.114 #<=fair
            conf_lim1 = 0.344 #<=good
            conf_lim2 = 0.849 #<=excellent
    
            if tot_conf<conf_lim0:
                REJECTED+=1
            elif tot_conf<conf_lim1:
                FAIRclusters+=1
            elif tot_conf<conf_lim2:
                GOODclusters+=1
            else:
                EXCELclusters+=1

            #Pbin getting a count depending on conf. percentile
            for i in range(1,11):
                if tot_conf<(i/10):
                    Pbin[i-1]+=1
                    break
            
    #Re-order        
    labels_arr = clusterSort(clusters, points)
    clusterOrder(clusters)                
    
    #SNPclusters
    ROWFORM(labels_arr,SNPclusters,df2_data)
    
    #SNRclusters
    SNRclusters = FAIRclusters+GOODclusters+EXCELclusters
    df2_data.append(SNRclusters)
    
    #NRFIclusters
    NRFIclusters = len(labels_arr)-1-REJECTED
    df2_data.append(NRFIclusters)
    
    #Rating for cluster
    df2_data.append(REJECTED)
    df2_data.append(FAIRclusters)
    df2_data.append(GOODclusters)
    df2_data.append(EXCELclusters)
    
    #Pbin to data
    df2_data = np.array(df2_data + Pbin)
    
    #Append to df2
    tempdf = pd.DataFrame([df2_data],columns=df2_labels)
    df2=df2.append(tempdf, ignore_index=True)
    #print(df2.head())
    
    #more memory
    del tempdf
    df2_data = []
    labels = []
    clusters = []
    points = []
    #print("Dm limit 2: " + str(dm_lim))
    #print("Old array length: " + str(len(points)))
    #print("New array length: " + str(len(points_new)))

#plt.hist(ratios,bins=10)
#Writing to file
df2.to_csv('condition_stats_14',index=False)

