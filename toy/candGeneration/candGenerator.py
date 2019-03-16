import numpy as np
import matplotlib as mpl
import glob, os
import warnings
from matplotlib import pyplot as plt
import math
import random

print(os.getcwd())

warnings.filterwarnings("ignore", category=mpl.cbook.mplDeprecation)
warnings.filterwarnings("ignore", category=RuntimeWarning)

freq = 0.334                    # F0 of survey
bandWidth = 64                  # Survey bandwidth

dmRangeBot = 100                # Lower possible DM for burst 
dmRangeTop = 900                # Upper possible DM for burst
dmWidthBot = 0                  # Lower possible DM width of a burst
dmWidthTop = 150                # Upper possible DM width of a burst
snRangeBot = 10                 # Lower possible peak SN for burst
snRangeTop = 40                 # Upper possible peak SN for burst
upperTime = 50                  # Upper possible time of detection for burst in seconds
upperDur = 600                  # Upper possible duration of burst in milliseconds
quantFrac = 0.8                 # Fraction over which the quantile method to calculate duration is used
fraction = 0.75                 # Probability of there being a point in a single DM step
noiseFraction = 0.1    # Fraction of noise events in a candidate

numBursts = 10                 # Number of bursts to simulate
bursts = []                     # All bursts are stored here, array of arrays
width = []

# Looping through the number of bursts to generate them
for k in range(numBursts):
    
    snArr = []              # All SN values of a burst
    dmArr = []              # All DM value of a burst
    tArr = []               # All times for the burst
    timeMid = np.random.uniform(1,upperTime)                # Average time of detection for a burst
    dmMid = np.random.uniform(dmRangeBot, dmRangeTop)       # Average DM of the burst
    peakSN = np.random.uniform(snRangeBot, snRangeTop)      # Peak SN of the burst

    # DM width of a burst
    dmWidth = 0
    while dmWidth <= 20:
        dmWidth = np.random.normal((dmWidthTop - dmWidthBot)/2,(dmWidthTop - dmWidthBot)/4)
    width.append(dmWidth)
    # Time duration of the burst
    Wms = 0
    while (Wms < 1):
        Wms = (np.random.gumbel(0.25,0.09)*80)**1.5     # Draws duration from this distribution
    timeRange = Wms/(2*quantFrac)       # Time range for the time distribution of the burst, centered around 0
    
    numPoints = 0       # Number of points in the burst
    while numPoints < 20:
        numPoints = int(np.random.gumbel(0.015, 0.06)*2000) # Draws the number of points from this distribution
    
    
    stDevSN = 2             # Standard deviation of the SN of the burst events
    stDevDM = dmWidth*0.1   # Standard deviation of the DM of the burst events
    start = -dmWidth/2          # Lower end of the DM range of the burst, currently centered around 0
    step = dmWidth/numPoints    # Size of each DM step when looping through the DM range
    count = 0                   # Countiong the number of events in the burst 
    
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
            dmArr.append(dmMid + dmTemp + devDM)    # Adds the actual DM value of the point to an array

            zeta = (6.91*10**-3)*bandWidth*(freq**-3)*(Wms**-1)*(dmTemp + devDM)        # Zeta-function
            snArr.append(math.pi**(1/2)*0.5*(zeta**-1)*math.erf(zeta)*(peakSN + devSN)) # SN value of the point, including deviation
            
            timeVar = timeMid + np.random.uniform(-timeRange,timeRange)/1000    # Time of detection of the points
            tArr.append(timeVar)    # Adds the time to the time array
        
            if tempVar1 <= noiseFraction*fraction:  # Generates the the noise points
                count += 1
                noiseDM = np.random.uniform((dmMid + start*1.1 - stDevDM*3), (dmMid + stDevDM*3 - start*1.1))
                noiseSN = np.random.uniform(0, peakSN*1.1)
                noiseTime = timeMid + np.random.uniform(-timeRange,timeRange)/1000
                
                dmArr.append(noiseDM)
                snArr.append(noiseSN)
                tArr.append(noiseTime)
        
    # Creates a numpy table of the burst, same format as the standard .dat files
    totArr = np.zeros([count, 3])
    totArr[:,0] = np.array(np.transpose(dmArr))
    totArr[:,1] = np.array(np.transpose(tArr))
    totArr[:,2] = np.array(np.transpose(snArr))

    bursts.append(totArr)
    

fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(len(bursts)):

    ax.scatter(bursts[i][:,1],bursts[i][:,0], s = 5)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.scatter(bursts[i][:,0], bursts[i][:,2], s = 5, label = str(width[i]))
    ax1.set_xlabel(str(max(bursts[i][:,1]) - min(bursts[i][:,1])))
    ax1.legend()
    
ax.set_xlabel("Time")
ax.set_ylabel("DM")
ax.set_xlim(0, 60)
ax.set_ylim(0, 1000)
ax.set_title("Cordes function")
plt.show()












