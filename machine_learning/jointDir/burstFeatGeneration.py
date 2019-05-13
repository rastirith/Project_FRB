import numpy as np
import matplotlib as mpl
import glob, os
import warnings
import math
import random
import pandas as pd
import sys
from bisect import bisect_left
from matplotlib import pyplot as plt
from scipy import special

warnings.filterwarnings("ignore", category=mpl.cbook.mplDeprecation)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from bandSim import bandCand
from noiseGenAlt import noiseGenerator

def rescale(arr, newMin, newMax):

    oldRange = (np.amax(arr) - np.amin(arr))  
    newRange = (newMax - newMin)  
    newValue = (((arr - np.amin(arr)) * newRange) / oldRange) + newMin

    return newValue

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


def cordes(Wms,SNratio):
    """Calculate the dm range spanned by band at value given by SNratio

    Keyword arguments:
    Wms -- the width (ms) property of signal under consideration
    SNratio -- the ratio of SN w.r.t. peak. Value at which the dm range is calculated.
    """
    freq = 0.334    # Centre frequency of survey
    bandWidth = 64  # Bandwidth of survey

    x = np.linspace(-1000,1000,10000)                           # Generates x-values for the cordes function
    zeta = (6.91*10**-3)*bandWidth*(freq**-3)*(Wms**-1)*x       # Zeta function in the cordes function 
    y = (math.pi**(1/2))*0.5*(zeta**-1)*special.erf(zeta)       # Values of cordes function, ranges between 0 and 1
    
    dm = x[np.nonzero(y > SNratio)]         # Array containing all dm values corresponding to y-values > SNratio
    dm_range = dm[-1] - dm[0]               # Theoretical allowed DM range for the current candidate
    
    return dm_range

def takeClosestNum(myList, myNumber):
    """Finds the closest value to numbers from a list defined by myList

    Keyword arguments:
    myList -- list of values that myNumber can be found close to
    myNumber -- values under consideration
    """
    pos = bisect_left(myList, myNumber)     # Bisection search to return an index
    if pos == 0:
        return myList[0]    # Before range of list return first value
    if pos == len(myList):
        return myList[-1]   # Outside range return last value
    before = myList[pos - 1]    # Value 1 index prior
    after = myList[pos]     # Value 1 index later
    if after - myNumber < myNumber - before:
       return after
    else:
       return before
 
def takeClosestArr(myList, myArr):
    """Finds the closest value of values in myArr to numbers from a list defined 
    by myList

    Keyword arguments:
    myList -- list of values that myArr can be found close to
    myArr -- array with values under consideration
    """
    tempArr = []
    for i in myArr:
        
        pos = bisect_left(myList, i)    # Bisection search to return and index
        if pos == 0:
            return myList[0]    # Before range of list return first value
        if pos == len(myList):
            return myList[-1]   # Outside range return last value
        before = myList[pos - 1]    # Value 1 index prior
        after = myList[pos]     # Value 1 index later
        if after - i < i - before:
           tempArr.append(after)
        else:
           tempArr.append(before)
           
    return tempArr
   
def timeGen(timeArr, dmArr, snArr, sdev = 1.44806):
    """Applies a slope in time-DM space of clusters found empirically

    Keyword arguments:
    timeArr -- time data of cluster
    dmArr -- DM data of cluster
    snArr -- SN data of cluster
    sdev -- Standard deviation in DM space of points around the slope 
    """
    slope = np.random.normal(-1636.53, 553.15)      # Slope of straight line in time-DM space, found empirically
    peakInd = np.argmax(snArr)                      # Index of the cluster peak in DM-SN space
    peakDM = dmArr[peakInd]                         # DM corresponding to SN peak
    peakTime = timeArr[peakInd]                     # Time corresponding to SN peak
    
    intercept = peakDM - slope*peakTime             # Intercept of straight line
    
    timeShift = ((np.random.normal(dmArr, sdev) - intercept)/slope) - peakTime  # Amount that time is shifted by to create slope
    timeArr = timeArr + timeShift   # New, shifted, time data

    return timeArr

# Make directory for files to be output and inputted if they don't exist
try:
    os.mkdir(os.getcwd()+"\sim_data\\") #output folder
except:
    pass

try: ####If changed to inject into a file will need error hadnling for an empty idirt
    os.mkdir(os.getcwd()+"\idir\\") #input folder 
except:
    pass

intention = input("Generate candidate bursts only (c), or generate training set (t)? ")

while intention != "c" and intention != "t":
    print("Invalid input, please indicated by entering either 'c'" + " or 't'.")
    intention = input("Generate candidate bursts only (c), or generate training set (t)? ")

numBursts = input("Enter number of candidates to simulate: ")

while True:
    try:
        numBursts = int(numBursts)
        break
    except:
        print("Invalid input, please indicate choice using an integer.")
        numBursts = input("Enter number of candidates to simulate: ")

if intention == "c" or intention == "t":
    folderName = input("Name of folder to be created: ")
    while folderName == "":
        print("Please enter a name of the folder to be created: ")
        folderName = input("Name of folder to be created: ")
    while True:
        try:    # Only creates folders if they don't already exist
            os.mkdir(os.getcwd() + '\sim_data\\' + folderName)
            break
        except:
            print("A folder with this name already exists. Please enter a different name.")
            folderName = input("Name of folder to be created: ")

if intention == "c":
    os.mkdir(os.getcwd() + '\sim_data\\' + folderName + '\\bursts')
    os.mkdir(os.getcwd() + '\sim_data\\' + folderName + '\\noise')
    
freq = 0.334                    # F0 of survey
bandWidth = 64                  # Survey bandwidth

dmRangeBot = 100                # Lower possible DM for burst 
dmRangeTop = 650                # Upper possible DM for burst
dmWidthBot = 0                  # Lower possible DM width of a burst
snRangeBot = 10                 # Lower possible peak SN for burst
snRangeTop = 40                 # Upper possible peak SN for burst
quantFrac = 0.8                 # Fraction over which the quantile method to calculate duration is used
fraction = 0.75                 # Probability of there being a point in a single DM step
noiseFraction = 0.15             # Fraction of noise events in a candidate

bursts = []                     # All bursts are stored here, array of arrays
counter = 0

#noiseFunctions = funcGenerator()


source_paths = []   # Array of file paths to be reviewed
shape_vals = []     # Array containing the shape feature values of the candidates
skew_vals = []      # Array containing the skewness feature values of the candidates
kurt_vals = []      # Array containing the kurtosis feature values of the candidates
kstest_vals = []    # Array containing the ks-test feature values of the candidates
class_vals = []     # Array containing the classification labels of the candidates

timer1 = []
timer2 = []
# Loops through all .dat files to store them in the 'source_paths' array
for file in glob.glob(os.getcwd() + '\idir\\' + "*.dat"):
    source_paths.append(file)

# Import dedispersion plan
df_ddp = pd.read_csv("dd_plan.txt")

# Setup array for step limits
dd_DM = np.array(df_ddp["DM_stop"])
dd_step = np.array(df_ddp["DM_step"])

# Constructing DM_poss array of possible DM values from ddp
DM_start = df_ddp["DM_start"]
DM_stop = df_ddp["DM_stop"]
DM_step = df_ddp["DM_step"]
DM_poss = [0.0] 

for i in range(len(DM_stop)):
    DM_range=DM_stop[i]-DM_start[i]
    num=round(DM_range/DM_step[i])
    for j in range(int(num)):
        DM_poss.append(round(DM_poss[-1]+DM_step[i],3))

print("\nGenerating candidates")

if intention == "t":
    typeVar = 0
else:
    typeVar = 0.5

while counter < numBursts:
    progressBar(counter, numBursts)
    
    upperTime = 50
    
    snArr = []              # All SN values of a burst
    dmArr = []              # All DM value of a burst
    tArr = []               # All times for the burst
    wArr = []
    labArr = []
    timeMid = np.random.uniform(1,upperTime)                # Average time of detection for a burst
    dmMid = np.random.uniform(dmRangeBot, dmRangeTop)       # Average DM of the burst
    peakSN = np.random.uniform(snRangeBot, snRangeTop)      # Peak SN of the burst
    SNratio = np.random.uniform(0.1,0.25)
    
    dmWidthTop = 0.2*dmMid

    Wms = 0
    while (Wms < 1):
        Wms = (np.random.gumbel(0.25,0.09)*80)**1.5     # Draws duration from this distribution
    timeRange = Wms/(2*quantFrac)       # Time range for the time distribution of the burst, centered around 0
    
    dmWidth = cordes(Wms,SNratio)
    
    numPoints = 0       # Number of points in the burst
    while numPoints < 20:
        numPoints = int(np.random.gumbel(0.015, 0.06)*2000) # Draws the number of points from this distribution
    
    stDevSN = 2             # Standard deviation of the SN of the burst events
    stDevDM = dmWidth*0.1   # Standard deviation of the DM of the burst events
    start = -dmWidth/2          # Lower end of the DM range of the burst, currently centered around 0
    step = dmWidth/numPoints    # Size of each DM step when looping through the DM range
    count = 0                   # Countiong the number of events in the burst 
    noiseProb = 1/5
    noiseVar = np.random.uniform(0,1)
    
    candType = 0
    if candType == 0:
        candData = bandCand()
        dmData = candData[0]
        snArr = candData[1]
        wArr = candData[2]
        if noiseVar <= noiseProb:
            #print(len(wArr))
            dataArr = noiseGenerator(0)
            rescaledDM = rescale(dataArr[0], np.amin(dmData), np.amax(dmData))
            
            dmData = np.concatenate((dmData, rescaledDM))
            snArr = np.concatenate((snArr, dataArr[1]))
            wArr = np.concatenate((wArr, np.full((len(dataArr[0])), np.random.normal(30,1.8))))
        
        dmData += dmMid
        for i in range(len(dmData)):
            dmFinal = takeClosestNum(DM_poss, dmData[i])
            dmArr.append(dmFinal)
            
            timeVar = (timeMid + np.random.uniform(-timeRange,timeRange)/1000)
            tArr.append(timeVar)
            
            labArr.append(1)
            
    elif candType == 1:
    
        # Loops through here to generate all possible points for the burst
        for i in range(numPoints + 1):
            
            tempVar1 = random.uniform(0,1)      # Randomises a number to determine whether a point should be generated or not
            if tempVar1 >= fraction:    # Does not generate a point
                continue
            else:                       # Does generate a point
                count += 1
                dmTemp = start + step*i     # DM for the point to be generated, relative to the DM centre of the burst
                if dmTemp == 0:     # Sorts the dm = 0 issue for the zeta function
                    dmTemp = 0.00000001
            
                devSN = np.random.normal(0,stDevSN)   # Deviation from the theoretical SN value for the current point
                devDM = np.random.normal(0,stDevDM)   # Deviation from the DM value for the current point
    
                dmFinal = takeClosestNum(DM_poss, dmMid + dmTemp + devDM)
                dmArr.append(dmFinal)   # Adds the actual DM value of the point to an array that has been "pixelated" to match the p-band data
    
                zeta = (6.91*10**-3)*bandWidth*(freq**-3)*(Wms**-1)*(dmTemp + devDM)        # Zeta-function
                snArr.append(math.pi**(1/2)*0.5*(zeta**-1)*math.erf(zeta)*(peakSN + devSN)) # SN value of the point, including deviation
                
                timeVar = (timeMid + np.random.uniform(-timeRange,timeRange)/1000)    # Time of detection of the points
                    
                tArr.append(timeVar)    # Adds the time to the time array that has been "pixelated" to match the p-band data
                wArr.append(np.random.normal(30,1.8))
                labArr.append(1)
            
                if tempVar1 <= noiseFraction*fraction:  # Generates the the noise points
                    count += 1
                    noiseDM = np.random.uniform((dmMid + start*1.1 - stDevDM*3), (dmMid + stDevDM*3 - start*1.1))
                    noiseSN = np.random.uniform(0, peakSN*0.8)
                    noiseTime = 0.000256*(round((timeMid + np.random.uniform(-timeRange,timeRange)/1000)/0.000256))
                    
                    dmArr.append(takeClosestNum(DM_poss, noiseDM))
                    snArr.append(noiseSN)
                    tArr.append(noiseTime)
                    wArr.append(np.random.normal(30,1.8))
                    labArr.append(0)
    
    if len(np.unique(dmArr)) < 4:
        continue
    
    counter += 1
    tArr = timeGen(tArr, dmArr, snArr)
        
    # Creates a numpy table of the burst, same format as the standard .dat files
    totArr = np.zeros([len(dmArr), 5])
    totArr[:,0] = np.array(np.transpose(dmArr))
    totArr[:,1] = np.array(np.transpose(tArr))
    totArr[:,2] = np.array(np.transpose(snArr))
    totArr[:,3] = np.array(np.transpose(wArr))
    
    """
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.scatter(totArr[:,1], totArr[:,0], s = 4)"""
    """
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.scatter(totArr[:,0], totArr[:,2], s = 4)
    ax2.set_xlabel("burst_cand_" + str(counter) + ".dat")"""
    """
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax3.scatter(totArr[:,2], totArr[:,3], s = 4)"""
    

    if intention == "c":
        totArr.reshape(-1)
        totArr[:,4] = np.array(np.transpose(labArr))
        fileName = "burst_cand_" + str(counter) + ".dat"     # Name of the class file to be created
        totArr.astype(np.float32).tofile(os.getcwd() + '\\sim_data\\' + folderName + "\\bursts\\" + fileName)
    else:
        totArr[:,4] = np.array(np.transpose(labArr))
        bursts.append(totArr) 

progressBar(numBursts,numBursts)               
counter = 0
print("\nGenerating non-candidates")

# NOISE
while (counter < numBursts) and ((intention == "t") or (intention == "c")):
    progressBar(counter, numBursts)
    
    upperTime = 50
    
    snArr = []              # All SN values of a burst
    dmArr = []              # All DM value of a burst
    tArr = []               # All times for the burst
    wArr = []
    labArr = []
    timeMid = np.random.uniform(1,upperTime)                # Average time of detection for a burst
    dmMid = np.random.uniform(dmRangeBot, dmRangeTop)       # Average DM of the burst
    peakSN = np.random.uniform(snRangeBot, snRangeTop)      # Peak SN of the burst
    
    dmWidthTop = 0.2*dmMid
    
    # DM width of a burst
    dmWidth = 0
    while dmWidth <= 20:
        dmWidth = np.random.normal((dmWidthTop - dmWidthBot)/2,(dmWidthTop - dmWidthBot)/4)
    # Time duration of the burst
    Wms = 0
    while (Wms < 1):
        Wms = np.random.gumbel(0.13,0.12)*50     # Draws duration from this distribution
    timeRange = Wms/(2*quantFrac)       # Time range for the time distribution of the burst, centered around 0
    
    np.random.seed()
    numArr = [
            np.random.randint(20,30),
            np.random.randint(70,90),
            np.random.randint(200,280),
            np.random.randint(450,700)
            ]
    numPoints = np.random.choice(numArr, 1, p = [0.45,0.15,0.15,0.25])[0]     # Number of points in the burst
    
    stDevSN = np.random.uniform(0.1,1)             # Standard deviation of the SN of the burst events
    stDevDM = np.random.uniform(0,4)   # Standard deviation of the DM of the burst events
    stDevSN = 0
    stDevDM = 0
    start = -dmWidth/2          # Lower end of the DM range of the burst, currently centered around 0
    step = dmWidth/numPoints    # Size of each DM step when looping through the DM range
    count = 0                   # Countiong the number of events in the burst 
    

    typeVar = np.random.uniform(0,1)
    side = np.random.uniform(0,1)
    botSN = np.random.uniform(0,peakSN-1)
    horSN = np.random.uniform(9,15)
    horBotSN = np.random.uniform(9,14)
    horStDevSN = np.random.uniform(0,0.5)
    count = 0
    pointNum = 0
    horLinOrd = np.random.uniform(0,1)
    
    funcVar = np.random.uniform(0,1)
    
    if funcVar > 0:
        #dmArr = np.linspace(-dmWidth/2, dmWidth/2, numPoints + 1)
        dataArr = noiseGenerator(1)
        tempDM = dataArr[0]
        snArr = dataArr[1]
        dmArr = takeClosestArr(DM_poss, tempDM)
        try:
            len(dmArr)
        except:
            continue
        
        for i in range(len(snArr)):
            
            timeVar = 0.000256*round((timeMid + np.random.uniform(-timeRange,timeRange)/1000)/0.000256)     # Time of detection of the points
            tArr.append(round(timeVar,6))    # Adds the time to the time array that has been "pixelated" to match the p-band data
            wArr.append(np.random.normal(30,1.8))
            labArr.append(0)
            
        #dataArr = noiseGenerator()
        #snArr = dataArr[0]
        #tempDM = dataArr[1] - dmWidth/2 + dmMid

        #dmArr = takeClosestArr(DM_poss, tempDM)
        
            
    tArr = timeGen(tArr, dmArr, snArr)
    counter += 1
    # Creates a numpy table of the burst, same format as the standard .dat files
    totArr = np.zeros([len(dmArr), 5])
    totArr[:,0] = np.array(np.transpose(dmArr))
    totArr[:,1] = np.array(np.transpose(tArr))
    totArr[:,2] = np.array(np.transpose(snArr))
    totArr[:,3] = np.array(np.transpose(wArr))
    """
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.scatter(totArr[:,1], totArr[:,0])
    """
    """
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.scatter(totArr[:,0], totArr[:,2], s = 4)"""
    
    if intention == "c":
        totArr.reshape(-1)
        totArr[:,4] = np.array(np.transpose(labArr))
        fileName = "noise_cand_" + str(counter) + ".dat"     # Name of the class file to be created
        totArr.astype(np.float32).tofile(os.getcwd() + '\\sim_data\\' + folderName + "\\noise\\" + fileName)
    else:
        totArr[:,4] = np.array(np.transpose(labArr))
        bursts.append(totArr)    

bursts = np.array(bursts)
progressBar(numBursts,numBursts)              
counter = 0
"""
if intention == "t":
    print("\nCalculating features")
    shape_vals, skew_vals, kurt_vals, kstest_vals, class_vals = featureFile(bursts)

    # Creates dataframe table containing all feature values as well as the classification labels for each cluster
    dataframe = pd.DataFrame({'Shape Feature': shape_vals,
                              'Skewness': skew_vals,
                              'Kurtosis': kurt_vals,
                              'KS-test stat': kstest_vals,
                              'Label': class_vals})
    
    dataframe.to_csv(os.getcwd() + '\\sim_data\\' + folderName + "\\" + "feature_table.csv", index = False)   # Writes dataframe to .csv file
    progressBar(numBursts,numBursts)"""

    

