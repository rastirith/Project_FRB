import numpy as np
import matplotlib as mpl
import glob, os
import warnings
from matplotlib import pyplot as plt
import math
import random
import pandas as pd
from bisect import bisect_left

warnings.filterwarnings("ignore", category=mpl.cbook.mplDeprecation)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Method to calculate and return the theoretical DM range span given a certain
# time duration/width and peak magnitude
def cordes(Wms,SNratio):
    freq = 1.732
    bandWidth = 336

    x = np.linspace(-1000,1000,10000)   # Generates x-values for the cordes function
    zeta = (6.91*10**-3)*bandWidth*(freq**-3)*(Wms**-1)*x       # Zeta function in the cordes function
    
    first = 0       # Variable indicating whether bottom or top of DM range has been found
    bot_dm = 0      # Value of the lower end of the DM range
    top_dm = 0      # Value of the upper end of the DM range
    for i in range(len(x)): # Loops through the x-values and calculates the cordes value in each step
        y = (math.pi**(1/2))*0.5*(zeta[i]**-1)*math.erf(zeta[i])
        if (y >= SNratio) and (first == 0): # First time the theoretical ratio goes above the actual ratio go in here
            bot_dm = x[i]                   # This x-value corresponds to the bottom DM value
            first = 1                       # Changes variable value to not go into this statement again
        if (y <= SNratio) and (first == 1): # First time the theoretical ratio goes below the actual ratio after bottom is found
            top_dm = x[i]                   # This x-value corresponds to the top DM value
            break                           # Values have been found, break out of loop
    dm_range = top_dm - bot_dm              # Theoretical allowed DM range for the current candidate
    return dm_range

# Finds the closest value to a myNumber in sorted myList 
def takeClosest(myList, myNumber):
    pos = bisect_left(myList, myNumber) #bisection search to return and index
    if pos == 0:
        return myList[0]    #before range of list return first value
    if pos == len(myList):
        return myList[-1]   #outside range return last value
    before = myList[pos - 1]    #valude 1 index prior
    after = myList[pos]     #value 1 index later
    if after - myNumber < myNumber - before:
       return after
    else:
       return before
   
def timeGen(timeArr, dmArr, snArr, sdev = 1.44806):
    slope = np.random.normal(-1636.53, 553.15) 
    peakInd = np.argmax(snArr)
    peakDM = dmArr[peakInd]
    peakTime = timeArr[peakInd]
    
    intercept = peakDM - slope*peakTime
    
    timeShift = ((np.random.normal(dmArr, sdev) - intercept)/slope) - peakTime
    timeArr = timeArr + timeShift

    return timeArr

freq = 1.732                    # F0 of survey
bandWidth = 336                # Survey bandwidth

dmRangeBot = 56                # Lower possible DM for burst 
dmRangeTop = 57                # Upper possible DM for burst
dmWidthBot = 0                  # Lower possible DM width of a burst
dmWidthTop = 150                # Upper possible DM width of a burst
snRangeBot = 100                 # Lower possible peak SN for burst
snRangeTop = 160                 # Upper possible peak SN for burst
upperTime = 50                  # Upper possible time of detection for burst in seconds
upperDur = 50                   # Upper possible duration of burst in milliseconds
quantFrac = 0.8                 # Fraction over which the quantile method to calculate duration is used
fraction = 0.75                 # Probability of there being a point in a single DM step
noiseFraction = 0.5             # Fraction of noise events in a candidate

numBursts = 5                 # Number of bursts to simulate
bursts = []                     # All bursts are stored here, array of arrays
width = []

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

# Looping through the number of bursts to generate them
for k in range(numBursts):
    
    snArr = []              # All SN values of a burst
    dmArr = []              # All DM value of a burst
    tArr = []               # All times for the burst
    wArr = []

    timeMid = np.random.uniform(1,upperTime)                # Average time of detection for a burst
    dmMid = np.random.uniform(dmRangeBot, dmRangeTop)       # Average DM of the burst
    peakSN = np.random.uniform(snRangeBot, snRangeTop)      # Peak SN of the burst
    SNratio = np.random.uniform(0.05,0.25)
    SNratio = 0.01
    #progressBar(counter, numBursts)
    
    noiseFraction = 0
    while (noiseFraction < 0):
        noiseFraction = np.random.normal(0.15,0.2)
        
    # Time duration of the burst
    Wms = 0
    while (Wms < dmMid/25):
        Wms = np.random.gumbel(0.13,0.12)*50     # Draws duration from this distribution
    timeRange = Wms/(2*quantFrac)       # Time range for the time distribution of the burst, centered around 0
    Wms = 0.192
    #Wms = 20
    dmWidth = cordes(16*Wms,SNratio)

    numPoints = 0       # Number of points in the burst
    while numPoints < 20:
        numPoints = int(np.random.gumbel(0.015, 0.06)*2000) # Draws the number of points from this distribution
    numPoints = 10000
    stDevSN = 2             # Standard deviation of the SN of the burst events
    stDevDM = dmWidth*0.1   # Standard deviation of the DM of the burst events
    start = -dmWidth/2          # Lower end of the DM range of the burst, currently centered around 0
    step = dmWidth/numPoints    # Size of each DM step when looping through the DM range
    count = 0                   # Countiong the number of events in the burst 
    
    for i in range(3):
        if i == 1:
            peakSN *= 0.5
            Wms= 2.176
            dmWidth = cordes(Wms,SNratio)
            stDevDM = dmWidth*0.1   # Standard deviation of the DM of the burst events
            start = -dmWidth/2          # Lower end of the DM range of the burst, currently centered around 0
            step = dmWidth/numPoints    # Size of each DM step when looping through the DM range
            SNratio = 0.05
        elif i > 1:
            peakSN *= 0.8
            Wms = 4.352
            dmWidth = cordes(Wms,SNratio)
            stDevDM = dmWidth*0.1   # Standard deviation of the DM of the burst events
            start = -dmWidth/2          # Lower end of the DM range of the burst, currently centered around 0
            step = dmWidth/numPoints    # Size of each DM step when looping through the DM range
            SNratio = 0.05
            
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
                
                dmFinal = takeClosest(DM_poss, dmMid + dmTemp + devDM)
                dmArr.append(dmFinal)    # Adds the actual DM value of the point to an array
    
                zeta = (6.91*10**-3)*bandWidth*(freq**-3)*(Wms**-1)*(dmTemp + devDM)        # Zeta-function
                snArr.append(math.pi**(1/2)*0.5*(zeta**-1)*math.erf(zeta)*(peakSN + devSN)) # SN value of the point, including deviation
                
                timeVar = (timeMid + np.random.uniform(-timeRange,timeRange)/1000)    # Time of detection of the points
                
                tArr.append(timeVar)
                wArr.append(Wms)
                width.append(i)
            
                if tempVar1 <= noiseFraction*fraction:  # Generates the the noise points
                    count += 1
                    noiseDM = np.random.uniform((dmMid + start*1.1 - stDevDM*3), (dmMid + stDevDM*3 - start*1.1))
                    noiseSN = np.random.uniform(0, peakSN*0.8)
                    noiseTime = 0.000256*(round((timeMid + np.random.uniform(-timeRange,timeRange)/1000)/0.000256))
                    
                    dmArr.append(takeClosest(DM_poss, noiseDM))
                    snArr.append(noiseSN)
                    tArr.append(noiseTime)
                    wArr.append(Wms)

    tArr = timeGen(tArr, dmArr, snArr)
    # Creates a numpy table of the burst, same format as the standard .dat files
    totArr = np.zeros([count, 4])
    totArr[:,0] = np.array(np.transpose(dmArr))
    totArr[:,1] = np.array(np.transpose(tArr))
    totArr[:,2] = np.array(np.transpose(snArr))
    totArr[:,3] = np.array(np.transpose(wArr))

    bursts.append(totArr)
    
bursts = np.array(bursts)
fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(len(bursts)):
    print(width[i])
    ax.scatter(bursts[i][:,1],bursts[i][:,0], s = 5)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.scatter(bursts[i][:,0][bursts[i][:,2] >= 8], bursts[i][:,2][bursts[i][:,2] >= 8], s = 5, cmap = "Set1", c = bursts[i][:,3][bursts[i][:,2] >= 8])
    ax1.set_xlabel(str((max(bursts[i][:,1]) - min(bursts[i][:,1]))*1000))
    #ax1.legend()

ax1.set_ylim(0,)
ax1.set_xlim(30,100)
ax.set_xlabel("Time")
ax.set_ylabel("DM")
ax.set_xlim(0, 60)
ax.set_ylim(0, 670)
ax.set_title("Cordes function")
plt.show()












