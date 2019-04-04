import numpy as np
import pandas as pd
import matplotlib as mpl
import glob, os, sys
import warnings
import math
import pickle
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn import preprocessing
from scipy.stats import skew, kurtosis
from matplotlib import pyplot as plt

warnings.filterwarnings("ignore", category=mpl.cbook.mplDeprecation)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Returns 'num' lowest elements of array 'arr' in a new array
def sort(arr, num):
    xsorted = np.sort(arr)[:num]
    return xsorted

def progressBar(value, endvalue, bar_length=20):
    
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    
    sys.stdout.write("\rPercent: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))

# Re-orders a shuffled labels_arr array so that the index of the array and the
# labels in the labels array has a q-1 relationship counting from q = 0.
# Label -1 (q = 0) corresponds to RFI and everything above it is a legitimate cluster
def clusterOrder(clusterArr):

    lab_arr = np.unique(clusterArr)     # Creates an array containing each unique value of the cluster label array
    for n in range(1, len(lab_arr)):    # Loops through all the possible labels counting from 0 to the number of unique clusters
        clusterArr = np.where(clusterArr == lab_arr[n], n - 1, clusterArr)
    return clusterArr


# Creates dataframe for file
def DF(path):
    axislabels = ["DM", "Time", "S/N", "Width"]
    Tfile = open(path,'r')
    data = np.fromfile(Tfile,np.float32,-1) 
    c = data.reshape((-1,4))
    df = pd.DataFrame(c,columns=axislabels)
    Tfile.close()
    return df

#working on crab data
def DF_crab(path):
    labels = ["Time", "DM", "Width", "S/N"]
    #reading in data files    
    df = pd.read_csv(path, delimiter="       ", header = 0, names = labels, engine='python')
    df["Time"] *=1/1000
    df = df[["DM", "Time", "S/N", "Width"]]
    return df

# Conducts the 2d KS-test on the SN-DM distribution and the theoretical cordes equation
def ks_cordes(dmArr,snArr,timeArr,peakDmMean):
    freq = 0.334
    bandWidth = 64
    cordes = []             # y-values of the theoretical cordes function
    peakSN = max(snArr)     # Value of the higher SN-bin of the data
    snFreqArr = []          # Probability frequency distribution for the data
    cordesFreqArr = []      # Probability frequency distribution for the theoretical function
    
    Wms = np.percentile(timeArr,75)-np.percentile(timeArr,25)   # Time width using the quantile method
    Wms = Wms*1000                                              # Must be in milliseconds
    dmScaled = dmArr - peakDmMean                               # Centers the data around DM = 0
    snRatios = (snArr + 9)/(peakSN + 9)                         # Ratios of the SN-values in relation to the peak
    
    x = np.linspace(min(dmScaled),max(dmScaled),2000)           # X-values for cordes function
    zeta = (6.91*10**-3)*bandWidth*(freq**-3)*(Wms**-1)*x       # Zeta function, see Cordes & M
    zeta[zeta == 0] = 0.000001
    
    # Calculates the y-values of the theoretical function
    for i in range(len(x)):
        temp_arr = []
        y = (math.pi**(1/2))*0.5*(zeta[i]**-1)*math.erf(zeta[i])
        frequency = int(y*100)
        temp_arr = [x[i]] * frequency
        cordesFreqArr.extend(temp_arr)

    altArr = []

    # Creates prob. freq. representation of the SN distribution
    for i in range(len(snRatios)):
        temp_arr = []
        frequency = int(snRatios[i]*100)      # Needs to be an integer, timed it by 1000 to reduce rounding errors, proportions still same
        altArr.append(frequency)
        temp_arr = [dmScaled[i]] * frequency   # Creates the corresponding number of elements and adds it to the array
        snFreqArr.extend(temp_arr)

    # Creates prob. freq. representation of the cordes func.
    for i in range(len(cordes)):
        temp_arr = []
        frequency = int(cordes[i]*100)
        temp_arr = [x[i]] * frequency
        cordesFreqArr.extend(temp_arr)
        
    statistic = stats.ks_2samp(snFreqArr,cordesFreqArr) #2D KS-test
    return statistic[0]

# Method to calculate and return the theoretical DM range span given a certain
# time duration/width and peak magnitude
def cordes(timeArr,peakSN):
    freq = 0.334
    bandWidth = 64
    cordes = []                 # Array containing all theoretical SN-values from the cordes function
    SNratio = 0.1    # Ratio of the peak to the cutoff point in the data
                                # Need to calculate the corresponding DM range for a reduction from 
                                # 1 (peak) to this ratio (tails)
    ratioMargin = 0 #0.05
    if (SNratio - ratioMargin) > 0:
        SNratio -= ratioMargin

    Wms = np.percentile(timeArr,100)-np.percentile(timeArr,0)   # Using the quantile method to calculate the width
    Wms = Wms*1000                                              # Width in ms
    
    x = np.linspace(-1000,1000,10000)   # Generates x-values for the cordes function
    zeta = (6.91*10**-3)*bandWidth*(freq**-3)*(Wms**-1)*x       # Zeta function in the cordes function
    
    first = 0       # Variable indicating whether bottom or top of DM range has been found
    bot_dm = 0      # Value of the lower end of the DM range
    top_dm = 0      # Value of the upper end of the DM range
    for i in range(len(x)): # Loops through the x-values and calculates the cordes value in each step
        y = (math.pi**(1/2))*0.5*(zeta[i]**-1)*math.erf(zeta[i])
        cordes.append(y)
        if (y >= SNratio) and (first == 0): # First time the theoretical ratio goes above the actual ratio go in here
            bot_dm = x[i]                   # This x-value corresponds to the bottom DM value
            top_dm = x[10000 - i]
            break
    dm_range = top_dm - bot_dm
    return dm_range

# Method to calculate and return the theoretical DM range span given a certain
# time duration/width and peak magnitude
def cordesAlt(widthArr,snArr, dmArr, ix):
    freq = 1.732
    bandWidth = 336
    cordes = []                 # Array containing all theoretical SN-values from the cordes function
                     
    a = np.nonzero(widthArr == np.unique(widthArr)[ix])[0]
    Wms = np.unique(widthArr)[ix]
    #colors = np.full((len(a)), ix)
    dmStart = np.amin(dmArr[a])
    dmStop = np.amax(dmArr[a])
    
    peakSNind = np.argmax(snArr)
    midDM = dmArr[peakSNind]
    wSNmax = np.amax(snArr)
    
    
    
    #Wms = np.linspace(dmStart,dmStop,10000)    # Using the quantile method to calculate the width
    x = np.linspace(-100,100,1000)                              # Generates x-values for the cordes function
    

    zeta = (6.91*10**-3)*bandWidth*(freq**-3)*(Wms**-1)*x      # Zeta function in the cordes function

    for i in range(len(x)): # Loops through the x-values and calculates the cordes value in each step
        y = (math.pi**(1/2))*0.5*(zeta[i]**-1)*math.erf(zeta[i])
        cordes.append(y)
    
    x += midDM
    #newY = snArr[a]/wSNmax
    #cordes -= 1-wSNmax
    
    #widthArr[a] = widthArr[a]/np.argmax(widthArr[a])
    #ax.scatter(dmArr[a], snArr[a], vmin = -1, alpha = 0.2, vmax = len(np.unique(widthArr)), label = str(np.unique(widthArr)[ix]), cmap = "gnuplot", c = colors, s = 25*((np.unique(widthArr)[ix])))
    
    #fig1 = plt.figure()
    #ax1 = fig1.add_subplot(111)
    ax.plot(x, cordes, color = "k", alpha = 0.15)
    #ax1.set_xlim(-0.05, 1.05)
        

def lineFit(timeArr, dmArr):
    reg_stats = stats.linregress(timeData,dmData)
    sum = 0
    
    for i in range(len(timeArr)):
        y = timeData[i]*reg_stats[0] + reg_stats[1]
        sum += (dmData[i] - y)**2
        
    sdev = (sum/(len(dmData) - 1))**0.5
    return reg_stats[0], reg_stats[1], sdev

clf = pickle.load(open("model.sav",'rb'))   # Loads the saved Random Forest model

counter = 0     # Counter that counts the candidate number reaching the feature loop

pos_array_mp3 = [
        26, 27, 28, 29, 22, 23, 24, 68, 69,
        66, 67, 63, 64, 57, 58, 61, 62, 59,
        60, 56, 55, 54, 53, 50, 49, 48, 47,
        45, 46, 43, 44, 39, 40, 41, 42, 34, 
        35, 36, 37, 38, 30, 31, 32, 33, 19,
        20, 21, 16, 17, 18, 11, 12, 13, 14, 
        15, 6, 2, 3, 4, 1
        ]

pos_array = pos_array_mp3

source_paths = []   # Array of file paths to be reviewed

#defining a constant scaling for the specific crab data
dmMAX = 1634.5
tMAX = 13.4701 
scaleDUMMY = [[0,0],[0,tMAX],[dmMAX,0],[dmMAX, tMAX]]
scaleDUMMY = np.array(scaleDUMMY)
scaler = preprocessing.MinMaxScaler()
scaler.fit(scaleDUMMY)

# Loops through all .dat files to store them in the 'source_paths' array
for file in glob.glob(os.getcwd() + '\crab_cands\\' + "*.spccl"):
    source_paths.append(file)

try:    # Only creates folders if they don't already exist
    os.mkdir(os.getcwd() + '\crab_cands\\' + "\\candidates")
    os.mkdir(os.getcwd() + '\crab_cands\\' + "\\empty")
except:
    pass

shape_vals = []     # Array containing the shape feature values of the candidates
sharp_vals = []     # Array containing the sharpness feature values of the candidates
skew_vals = []      # Array containing the skewness feature values of the candidates
kurt_vals = []      # Array containing the kurtosis feature values of the candidates
kstest_vals = []    # Array containing the ks-test feature values of the candidates
class_vals = []     # Array containing the classification labels of the candidates

#colour cheating
colours = ["r","g","b","y","k"]*10

slopes = []
sdev_stats = []

y = 5

# Loops through the whole file space defined by 'source_paths'
for i in range(26,27): 
    #print(i)
    #progressBar(i,y)
    
    fileSize = os.path.getsize(source_paths[i])/1024000
    path_index = i      # Current path index in the source_paths array
    
    file = source_paths[path_index]     # Setting which file to open
    df = DF_crab(file) # Creates dataframe from the .dat file
    orig_X = np.array(df)   # Sets dataframe as a Numpy array
    
    X_db = np.array(df.drop(columns=['Width', 'S/N']))  # Drops width and S/N data for DBScan to run on the DM - Time space
    X = np.array(df)
    
    # Sorts the data points by DM
    points_db = X_db[X_db[:,0].argsort()]
    points = X[X[:,0].argsort()]
    """
    # Lower DM limit below which the DBScan is now run as it is most likely not extragalactic
    # Speeds up the DBScan runtime
    dm_lim = 0.03*max(points_db[:,0])
    points_new = np.array(points_db[points_db[:,0] > dm_lim])
    """
    X_scaled = scaler.transform(points_db) # Rescales the data so that the x- and y-axes get ratio 1:1
    if len(X_scaled)>10000:
        continue
    
    X_scaled[:,1] = 3*X_scaled[:,1] #fixing this time overlap with a stretch
    
    xeps = 0.01     # Radius of circle to look around for additional core points
    xmin = 5  # Number of points within xeps for the point to count as core point
    
    clusters = DBSCAN(eps=xeps, min_samples = xmin).fit_predict(X_scaled)   # Clustering algorithm, returns array with cluster labels for each point
    
    # Re-inserts bottom points with labels -1 for RFI
    length = len(points) - len(clusters)
    clusters = np.insert(clusters,0,np.full(length,-1))
    
    # Adds column to the points arrays for cluster label
    newArr = np.column_stack((points, clusters[np.newaxis].T))
    
    # Re-order
    newArr[:,4] = clusterOrder(newArr[:,4])
    """
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    
    ax1.scatter(newArr[:,1], newArr[:,0], s = 6,cmap = "Dark2" , c = newArr[:,4], vmin = -1)
    ax1.set_title(str(i))
    #ax1.set_xlabel(str(timeDiff))
    
    plt.show()
    """      
    
    
    
    
    
    """
    # Noise condition for Nevents<Nmin => noise
    N_min = 10# Tuneable number (set to match Karako)
    labels = np.unique(newArr[:,4])
    for q in range(1,len(labels)):
        label = labels[q]
        labSlice = np.extract(newArr[:,4] == label, newArr[:,4])
        if (len(labSlice) < N_min):        # Gives points of clusters with less than N_min events the RFI lable of -1
            newArr[:,4] = np.where(newArr[:,4] == label, -1, newArr[:,4])
    """
    # Re-order    
    newArr[:,4] = clusterOrder(newArr[:,4])
    
    # Break
    dm_lim = 20         # Increased DM-limit to check for clusters with 'fraction' of their points below this limit
    fraction = 0.05     # Size of the fraction of points allowed in below 'dm_lim'

    # Condition that sets all clusters with 'fraction' of its points below dm_lim to also be classified as RFI
    labels = np.unique(newArr[:,4])
    for q in range(1,len(labels)):
        label = labels[q]
        labSlice = np.extract(newArr[:,4] == label, newArr[:,4])
        num_temp = int(round(fraction*len(labSlice),0))    # Calculates how many points of a label a certain fraction corresponds to and rounds to nearest integer
        temp = sort(newArr[newArr[:,4] == label][:,0],num_temp)                # Returns the 'num_temp' lowest dms in labels_arr[q]
        if ((len(temp) > 0) and (max(temp) < dm_lim)):          # If the highest number in temp is below dm_lim then so is the rest in 'temp'
            newArr[:,4] = np.where(newArr[:,4] == label, -1, newArr[:,4])

    # Condition that sets all points with dm below dm_lim to be classified as RFI
    newArr[:,4][newArr[:,0] < dm_lim] = -1

    # Re-order    
    newArr[:,4] = clusterOrder(newArr[:,4])
    
    # Burst duration condition
    labels = np.unique(newArr[:,4])
    for q in range(1,len(labels)):
        label = labels[q]
        upper = np.quantile(newArr[newArr[:,4] == label][:,1], 0.8)
        lower = np.quantile(newArr[newArr[:,4] == label][:,1], 0.2)
        if (upper - lower) >= 1:            # If the time between the quantiles is longer than 1s set cluster to RFI
            newArr[:,4] = np.where(newArr[:,4] == label, -1, newArr[:,4])
    
    # Re-order    
    newArr[:,4] = clusterOrder(newArr[:,4])
    
    ANY = []            ###TESTING
    least_acc = []      # Contains the points of all candidates classified as 'least acceptable'
    good = []           # Contains the points of all candidates classified as 'good'
    excellent = []      # Contains the points of all candidates classified as 'excellent'
    labelNumber = 0     # Label number of clusters in same files, used to separate multiple clusters of equal classification

    # Loops through all remaining clusters to exclude further clusters, calculate feature values, and classify them using Random Forest
    labels = np.unique(newArr[:,4])
    
    fig1=plt.figure()            
    ax1 = fig1.add_subplot(111)
    ax1.set_title("path"+str(path_index))
    ax1.scatter(points[:,1], points[:,0], s = 6, color = '0.7', vmin = -1)
    
    for q in range(1,len(labels)):
        
        
        label = labels[q]
        dmData = newArr[newArr[:,4] == label][:,0]
        timeData = newArr[newArr[:,4] == label][:,1]
        snData = newArr[newArr[:,4] == label][:,2]
        widthData = newArr[newArr[:,4] == label][:,3]
        
        signalToDm = list(zip(dmData, snData))  # Array containing DM - SN data
        signalToDm = np.array(signalToDm)
        min_val = min(signalToDm[:,1])                                  # Finds the lowest SN-value in the candidate array

        scaled_signal = preprocessing.MinMaxScaler().fit_transform(signalToDm)  # Preprocesses signalToDM to used for sharpness condition

        # Sets y = 0 for visualisation
        for i in range(len(signalToDm[:,1])):
            signalToDm[:,1][i] = signalToDm[:,1][i] - min_val
            
        # Splitting into chunks of equal number of events in each
        split_param = 7 # Number of chunks to be split into
        dummy = np.array_split(signalToDm,split_param)
        
        meanSN = []     # Contains mean SN value of each chunk 
        meanDM = []     # Contains mean DM value of each chunk
       
        # Loops through the chunks, calculates the relevant mean values and puts them into the appropriate arrays
        for i in range(len(dummy)):
            tempSN = np.mean(dummy[i][:,1])
            tempDM = np.mean(dummy[i][:,0])
            meanSN.append(tempSN)
            meanDM.append(tempDM)
        
        max_val = max(meanSN + min_val) # Array containing the real (not reduced) peak SN values in each chunk
        max_ind = np.argmax(meanSN)     # Finds the index for the highest SN bin value
        peakMeanDm = meanDM[max_ind]    # The corresponding DM for this bin value
        
        upper_dm_range = cordes(timeData, meanSN[max_ind]) # The theoretically widest allowed DM range of candidate
        dm_diff = max(dmData) - min(dmData)  # The DM range spanned by the actual data
        
        # Condition that excludes the candidates with SN peaks too far from the centre, as well as candidates spanning a range that is too wide
        # Using 2 times the upper_dm_range to use a large safety margin
        if (max_ind > 4) or (max_ind < 2) or (dm_diff > 2.5*upper_dm_range):  
            newArr[:,4] = np.where(newArr[:,4] == label, -1, newArr[:,4])
            #print("here")
        # All candidates that haven't been excluded thus far go in here
        else:
            counter += 1        # Final candidate number reaching this point
            freq_arr = []       # Probability frequency distribution representation of the DM - SN data
            weight_1 = -1       # Score weight if ratio is less than 1 but more than 1 - check_1
            weight_2 = -0.3     # Score weight if ratio is less than 1 - check_1, but more than 1 - check_2
            weight_3 = 1        # Score weight if ratio is less than 1 - check_2
            weight_4 = -1       # Score weight if ratio is more than 1
            check_1 = 0.075
            check_2 = 0.15
            score = [0,1.3,2.5,2.5]                         # Scoring system where one index step corresponds to one step from peak bin
            max_score = 2*(score[0] + score[1] + score[2])  # Maximum possible score
            rating = 0  # Rating score after weight and scores have been applied
            
            for i in range(max_ind - 1, -1, -1):                # Loops through all bins from the peak bin moving to the left
                ratio=meanSN[i]/meanSN[i+1]                     # Ratio of the next bin to the previous bin
            
                if ((ratio>=(1-check_1)) and (ratio<=1)):
                    rating += weight_1*score[max_ind-(i+1)]
                elif ((ratio>=(1-check_2)) and (ratio<=1)):
                    rating += weight_2*score[max_ind-(i+1)]
                elif (ratio<=1):
                    rating += weight_3*score[max_ind-(i+1)]
                else:
                    rating += weight_4*score[max_ind-(i+1)]
                            
            for i in range((max_ind+1),split_param):            # Loops through all bins from the peak bin moving to the right
                ratio=meanSN[i]/meanSN[i-1]

                if ((ratio>=(1-check_1)) and (ratio<=1)):
                    rating += weight_1*score[i-max_ind-1]
                elif ((ratio>=(1-check_2)) and (ratio<=1)):
                    rating += weight_2*score[i-max_ind-1]
                elif ratio <=1:
                    rating += weight_3*score[i-max_ind-1]
                else:
                    rating += weight_4*score[i-max_ind-1]
                    
            # Exception case where rating is less than 0, sets rating to 0 if this happens
            if rating < 0:
                rating = 0

            # Converts the S/N-DM plot into a probability frequency plot
            # Instead of each point in DM space having a corresponding S/N y-value
            # there will be an array containing a number of DM elements proportional to its S/N value
            normal_snRatios = (signalToDm[:,1] + 9)/(max(signalToDm[:,1]) + 9)
            for i in range(len(signalToDm)):
                temp_arr = []
                frequency = int((normal_snRatios[i])*1000)      # Needs to be an integer, timed it by 1000 to reduce rounding errors, proportions still same
                temp_arr = [signalToDm[i][0]] * frequency   # Creates the corresponding number of elements and adds it to the array
                freq_arr.extend(temp_arr)

            # FEATURES
            shape_conf = rating/max_score                       # Shafe conf feature
            skewness = skew(freq_arr, axis = 0)                 # Skewness feature
            kurt = kurtosis(freq_arr, axis = 0, fisher = True)  # Kurtosis feature
            ks_stat = ks_cordes(signalToDm[:,0],signalToDm[:,1],timeData,meanDM[max_ind])     # KS feature
            
            # Adds the feature values to the corresponding arrays
            shape_vals.append(shape_conf)
            skew_vals.append(skewness)
            kurt_vals.append(kurt)
            kstest_vals.append(ks_stat)
            
            features = [shape_conf, 0, skewness, kurt, ks_stat]       # Arrays of features to be ran through Random Forest model
            results = [[0,0]] #clf.predict_proba([features])                             # Runs Random Forest model on features arrays

            
            timeDiff = upper - lower
            """
            if (results[0][1] > 0.8):       # If the final rating from Random Forest is more than 0.8 goes in here
                labNumArray = np.full((len(dmData), 1), labelNumber)
                tempCandArr = np.column_stack((newArr[newArr[:,4] == label], labNumArray))
                excellent.extend(tempCandArr)
                labelNumber += 1
                
            elif (results[0][1] > 0.65):    # If the final rating from Random Forest is more than 0.65 but less than 0.8, goes in here
                labNumArray = np.full((len(dmData), 1), labelNumber)
                tempCandArr = np.column_stack((newArr[newArr[:,4] == label], labNumArray))
                good.extend(tempCandArr)
                labelNumber += 1
                
            elif (results[0][1] > 0.5):     # If the final rating from Random forest is more than 0.5 but less than 0.65, goes in here
                labNumArray = np.full((len(dmData), 1), labelNumber)
                tempCandArr = np.column_stack((newArr[newArr[:,4] == label], labNumArray))
                least_acc.extend(tempCandArr)
                labelNumber += 1
            """
            #elif(results[0][1] == 0):     #testing skip of RF
            labNumArray = np.full((len(dmData), 1), labelNumber)
            tempCandArr = np.column_stack((newArr[newArr[:,4] == label], labNumArray))
            ANY.extend(tempCandArr)
            
            snNorm = snData/(max(snData))
            print("count: ", counter)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            freq = 1.732
            bandWidth = 336
            
            if counter == 1:
                fSNarr = []
                sSNarr = []
                tSNarr = []
                foSNarr = []
                pSNarr = []
                logSN = []
                logWidth = np.log([0.192, 2.176, 4.352, 8.704, 17.408])
                logSN.append(np.log(169))
                
                dmRange = np.arange(-30, +30, 0.1)
                #2.176
                inds1 = np.nonzero((widthData == 2.176) & (snData > 20))
                sliceDM1 = dmData[inds1]
                sliceSN1 = snData[inds1]
                fRange = max(sliceDM1) - min(sliceDM1)
                zeta = (6.91*10**-3)*bandWidth*(freq**-3)*(2.176**-1)*(fRange/2)
                fRat = (math.pi**(1/2)*0.5*(zeta**-1)*math.erf(zeta))
                fPeak = np.mean(sliceSN1)/fRat
                zeta = (6.91*10**-3)*bandWidth*(freq**-3)*(2.176**-1)*(dmRange)
                zeta[zeta == 0] = 0.000001
                for i in range(len(dmRange)):
                    fSNarr.append((math.pi**(1/2)*0.5*(zeta[i]**-1)*math.erf(zeta[i]))*fPeak)
                logSN.append(np.log(max(sliceSN1)))
                
                
                #4.352
                inds2 = np.nonzero((widthData == 4.352) & (snData > 20))
                sliceDM2 = dmData[inds2]
                sliceSN2 = snData[inds2]
                sRange = max(sliceDM2) - min(sliceDM2)
                zeta = (6.91*10**-3)*bandWidth*(freq**-3)*(4.352**-1)*(sRange/2)
                sRat = (math.pi**(1/2)*0.5*(zeta**-1)*math.erf(zeta))
                sPeak = np.mean(sliceSN2)/sRat
                zeta = (6.91*10**-3)*bandWidth*(freq**-3)*(4.352**-1)*(dmRange)
                zeta[zeta == 0] = 0.000001
                for i in range(len(dmRange)):
                    sSNarr.append((math.pi**(1/2)*0.5*(zeta[i]**-1)*math.erf(zeta[i]))*sPeak)
                logSN.append(np.log(max(sliceSN2)))
                
                #8.704
                inds3 = np.nonzero((widthData == 8.704) & (snData > 15) & (dmData > 47))
                sliceDM3 = dmData[inds3]
                sliceSN3 = snData[inds3]
                tRange = max(sliceDM3) - min(sliceDM3)
                zeta = (6.91*10**-3)*bandWidth*(freq**-3)*(8.704**-1)*(tRange/2)
                tRat = (math.pi**(1/2)*0.5*(zeta**-1)*math.erf(zeta))
                tPeak = np.mean(sliceSN3)/tRat
                zeta = (6.91*10**-3)*bandWidth*(freq**-3)*(8.704**-1)*(dmRange)
                zeta[zeta == 0] = 0.000001
                for i in range(len(dmRange)):
                    tSNarr.append((math.pi**(1/2)*0.5*(zeta[i]**-1)*math.erf(zeta[i]))*tPeak)
                logSN.append(np.log(max(sliceSN3)))
                
                #17.408
                inds4 = np.nonzero((widthData == 17.408) & (snData > 10))
                sliceDM4 = dmData[inds4]
                sliceSN4 = snData[inds4]
                foRange = max(sliceDM4) - min(sliceDM4)
                zeta = (6.91*10**-3)*bandWidth*(freq**-3)*(17.408**-1)*(20)
                foRat = (math.pi**(1/2)*0.5*(zeta**-1)*math.erf(zeta))
                foPeak = np.mean(sliceSN4)/foRat
                zeta = (6.91*10**-3)*bandWidth*(freq**-3)*(17.408**-1)*(dmRange)
                zeta[zeta == 0] = 0.000001
                for i in range(len(dmRange)):
                    foSNarr.append((math.pi**(1/2)*0.5*(zeta[i]**-1)*math.erf(zeta[i]))*foPeak)
                logSN.append(np.log(max(sliceSN4)))  
                    
                zeta = (6.91*10**-3)*bandWidth*(freq**-3)*(0.192**-1)*(dmRange)
                zeta[zeta == 0] = 0.000001
                for i in range(len(dmRange)):
                    pSNarr.append((math.pi**(1/2)*0.5*(zeta[i]**-1)*math.erf(zeta[i]))*169.632)
                
                print("2.176ms range: ", fRange , "\n4.352ms range: ", sRange, "\n8.704ms range: ", tRange, "\n17.408ms range: ", foRange)
                print("2.176ms ratio: ", fRat , "\n4.352ms ratio: ", sRat, "\n8.704ms ratio: ", tRat, "\n17.408ms ratio: ", foRat)
                print("2.176ms factor: ", fPeak , "\n4.352ms factor: ", sPeak, "\n8.704ms factor: ", tPeak, "\n17.408ms factor: ", foPeak)
                
                
                
                inds1 = np.nonzero((widthData == 2.176))
                sliceDM1 = dmData[inds1]
                sliceSN1 = snData[inds1]
                inds2 = np.nonzero((widthData == 4.352))
                sliceDM2 = dmData[inds2]
                sliceSN2 = snData[inds2]
                inds3 = np.nonzero((widthData == 8.704))
                sliceDM3 = dmData[inds3]
                sliceSN3 = snData[inds3]
                inds4 = np.nonzero((widthData == 17.408))
                sliceDM4 = dmData[inds4]
                sliceSN4 = snData[inds4]
                
                for m in range(1,11):
                    temp = []
                    zeta = (6.91*10**-3)*bandWidth*(freq**-3)*((0.5*m)**-1)*(dmRange)
                    zeta[zeta == 0] = 0.000001
                    for i in range(len(dmRange)):
                        temp.append((math.pi**(1/2)*0.5*(zeta[i]**-1)*math.erf(zeta[i]))*169.632)
                    xRange = dmRange + 57
                    ax.plot(xRange, temp, c = "k", alpha = 0.2)

                dmRange += 57
                #logSN = []
                for i in reversed(range(len(np.unique(widthData)))):
                    a = np.nonzero(widthData == np.unique(widthData)[i])[0]
                    colors = np.full((len(a)), i)
                    
                    #cordesAlt(widthData, snNorm, dmData, i)
                    #logSN.append(np.log(max(snData[a])))
                    #print(logSN)
                    b = ax.scatter(dmData[a], snData[a], vmin = -1, vmax = len(np.unique(widthData)), label = str(np.unique(widthData)[i]), cmap = "gnuplot", c = colors, s = 6)#, s = 25*((np.unique(widthData)[i])))
                    
                    altY = snNorm[a]/np.amax(snNorm[a])
                    #print(str(np.unique(widthData)[i]))
                    #cordesAlt(widthData, altY, dmData, i)
                    
                
                
                
                fig2 = plt.figure()
                ax2 = fig2.add_subplot(111)
                ax2.scatter(widthData, snData)
                
                slope = stats.linregress(logWidth,logSN)[0]
                intercept = stats.linregress(logWidth,logSN)[1]
                
                x = np.log(np.arange(0.180, 80, 0.001))
                #print(len(x))
                y = np.exp(slope*x)*np.exp(intercept)
                #print(len(y))
                x = np.exp(x)
                #print(y)
                #y = slope*logWidth + intercept
                
                ax2.plot(x, y, alpha = 0.5)
                
            ax.scatter(sliceDM1, sliceSN1, c = "g", s = 6)
            ax.scatter(sliceDM2, sliceSN2, c = "c", s = 6)
            ax.scatter(sliceDM3, sliceSN3, c = "g", s = 6)
            ax.scatter(sliceDM4, sliceSN4, c = "c", s = 6)
                    
            ax.plot(dmRange, fSNarr, c = "k", alpha = 0.3)
            ax.plot(dmRange, sSNarr, c = "k", alpha = 0.3)
            ax.plot(dmRange, tSNarr, c = "k", alpha = 0.3)
            ax.plot(dmRange, foSNarr, c = "k", alpha = 0.3)
            ax.plot(dmRange, pSNarr, c = "k", alpha = 0.3)        
            
            ax.figure.colorbar(b, ax=ax)
            ax.set_xlabel("DM")
            ax.set_ylabel("SN")
            ax.set_title(str(counter))
            #ax.set_xlim(np.amin(dmData),np.amax(dmData))
            #ax.set_ylim(-0.05,1.05)
            #ax.legend()

            plt.show()
            
            if (counter in pos_array):
                #reg_stats = stats.linregress(timeData,dmData)
                fit = lineFit(timeData, dmData)
                #sdev_stat =
                slopes.append(fit[0])
                sdev_stats.append(fit[2])
            
            #ax1.scatter(points[:,1], points[:,0], s = 6, color = '0.7', vmin = -1)
            
            ax1.scatter(timeData, dmData, s = 6, c = colours[labelNumber], label = counter, vmin = -1)
            
            ax1.set_xlabel(str(timeDiff))
            ax1.legend()
            #plt.show()
            #print((dmData))
                
            
            """
            # Adds the correct class label for the cluster to the class_vals array.
            # 1 = candidate, 0 = no candidate
            if (counter in pos_array):
                #reg_stats = stats.linregress(timeData,dmData)
                fit = lineFit(timeData, dmData)
                #sdev_stat =
                slopes.append(fit[0])
                sdev_stats.append(fit[2])"""
                
                
            labelNumber += 1
    
    
    # Re-order    
    newArr[:,4] = clusterOrder(newArr[:,4])
    
    ANY = np.array(ANY)
    least_acc = np.array(least_acc)
    good = np.array(good)
    excellent = np.array(excellent)
    
    total = []
    if_var = 0
    
    # Adds a column to the numpy array with an integer corresponding to the cluster classification
    # 3 = excellent, 2 = good, 1 = least acceptable
    if (len(excellent) > 0):
        excellent_c = np.full((len(excellent),7), 3, dtype = float)
        excellent_c[:,:-1] = excellent
        total.extend(excellent_c)
        if_var += 1
    if (len(good) > 0):
        good_c = np.full((len(good),7), 2, dtype = float)
        good_c[:,:-1] = good
        total.extend(good_c)
        if_var += 1
    if (len(least_acc) > 0):
        least_acc_c = np.full((len(least_acc),7), 1, dtype = float)
        least_acc_c[:,:-1] = least_acc
        total.extend(least_acc_c)
        if_var += 1
    if (len(ANY) > 0):
        ANY_c = np.full((len(ANY),7), 0, dtype = float)
        ANY_c[:,:-1] = ANY
        total.extend(ANY_c)
        if_var += 1
        
    
    total = np.array(total)         # Sets the total array containing feature values, class labels, and cluster numbers to Numpy array
    """
    if len(total) > 0:              # If the 'total' array contains candidates then go in here
        dataframe = pd.DataFrame({'DM': total[:,0],         # Creates a dataframe with feature values, class labels, and cluster numbers
                                  'Time': total[:,1],
                                  'S/N': total[:,2],
                                  'Width': total[:,3],
                                  'Class': total[:,6],
                                  'Cluster Number': total[:,5]})
        
        new_name = source_paths[path_index].split("crab_cands\\")[1].replace('.spccl','_c.csv')               # Name of the class file to be created
        dataframe.to_csv(os.getcwd() + '\\crab_cands\\' + "\\candidates\\" + new_name, index = False) # Saves csv file of the above dataframe
        new_path = os.getcwd() + '\\crab_cands\\' + "\\candidates\\" + source_paths[path_index].split("crab_cands\\")[1]  # Path to the created file
        
        if (os.path.isfile(new_path)) == False:     # Moves the .dat file to the 'candidates' directory only if there isn't already an
            os.rename(file, new_path)               # identical .dat file there.
            
    else:                           # If there are no candidates then move the .dat file to the 'empty' directory
        new_path = os.getcwd() + '\\crab_cands\\' + "\\empty\\" + source_paths[path_index].split("crab_cands\\")[1]
        if (os.path.isfile(new_path)) == False:
            os.rename(file, new_path)"""
progressBar(1,1)
slopes = np.array(slopes)/1000
#sdev_stats = np.array(sdev_stats)/1000
print("\n")# + str(slopes)) 
#print(str(sdev_stats))    
print("avg: " + str(np.round(np.mean(slopes),5)) + " DM/ms")
print("sdev: " + str(np.round(np.mean(sdev_stats),5)) + " DM")
print("sdev slopes: " + str(np.round(np.std(slopes),5)))




         