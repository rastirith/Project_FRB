import numpy as np
import matplotlib as mpl
import glob, os
import warnings
from matplotlib import pyplot as plt
import math
import random
import pandas as pd
import sys
from bisect import bisect_left


# Creates dataframe for file
def DF(path):
    axislabels = ["DM", "Time", "S/N", "Width"]
    Tfile = open(path,'r')
    data = np.fromfile(Tfile,np.float32,-1) 
    c = data.reshape((-1,4))
    df = pd.DataFrame(c,columns=axislabels)
    Tfile.close()
    return df

def progressBar(value, endvalue, bar_length=20):
    
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    
    sys.stdout.write("\rPercent: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))  
    
# Method to calculate and return the theoretical DM range span given a certain
# time duration/width and peak magnitude
def cordes(Wms,SNratio):
    freq = 0.334
    bandWidth = 64

    x = np.linspace(-1000,1000,10000)   # Generates x-values for the cordes function
    zeta = (6.91*10**-3)*bandWidth*(freq**-3)*(Wms**-1)*x       # Zeta function in the cordes function
    
    #first = 0       # Variable indicating whether bottom or top of DM range has been found
    bot_dm = 0      # Value of the lower end of the DM range
    top_dm = 0      # Value of the upper end of the DM range
    for i in range(len(x)): # Loops through the x-values and calculates the cordes value in each step
        y = (math.pi**(1/2))*0.5*(zeta[i]**-1)*math.erf(zeta[i])
        if (y >= SNratio): # First time the theoretical ratio goes above the actual ratio go in here
            bot_dm = x[i]                   # This x-value corresponds to the bottom DM value
            top_dm = x[10000 - i]
            break
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


warnings.filterwarnings("ignore", category=mpl.cbook.mplDeprecation)
warnings.filterwarnings("ignore", category=RuntimeWarning)

intention = input("Plot bursts (s), generate candidate bursts (g), or generate training set (t)? ")

while intention != "s" and intention != "g" and intention != "t":
    print("Invalid input, please indicated by entering either 's'," + "'g', " + "or 't'.")
    intention = input("Simulate plots (s), or generate data to use (g)? ")

numBursts = input("Enter number of candidates to simulate: ")

while True:
    try:
        numBursts = int(numBursts)
        break
    except:
        print("Invalid input, please indicate choice using an integer.")
        numBursts = input("Enter number of candidates to simulate: ")

if intention == "g" or intention == "t":
    folderName = input("Name of folder to be created: ")
    while folderName == "":
        print("Please enter a name of the folder to be created: ")
        folderName = input("Name of folder to be created: ")
    while True:
        try:    # Only creates folders if they don't already exist
            os.mkdir(os.getcwd() + '\odir\\' + folderName)
            break
        except:
            print("A folder with this name already exists. Please enter a different name.")
            folderName = input("Name of folder to be created: ")
"""          
if intention == "g":
    fileOrBurst = input("Generate files with inserted burst (i), burst only (b), or features file of bursts (f)? " )
    while fileOrBurst != "i" and intention != "b" and intention != "f":
        print("Invalid input, please indicated by entering either 'i'," + "'b', " + "or 'f'.")
        fileOrBurst = input("Generate files with inserted burst (i), burst only (b), or features file of bursts (f)? " )"""
        
freq = 0.334                    # F0 of survey
bandWidth = 64                  # Survey bandwidth

dmRangeBot = 100                # Lower possible DM for burst 
dmRangeTop = 650                # Upper possible DM for burst
dmWidthBot = 0                  # Lower possible DM width of a burst
snRangeBot = 10                 # Lower possible peak SN for burst
snRangeTop = 40                 # Upper possible peak SN for burst
quantFrac = 0.8                 # Fraction over which the quantile method to calculate duration is used
fraction = 0.75                 # Probability of there being a point in a single DM step

bursts = []                     # All bursts are stored here, array of arrays
counter = 0

source_paths = []   # Array of file paths to be reviewed

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
while counter < numBursts:
    #progressBar(counter, numBursts)
    path_index = int(round(np.random.uniform(0,len(source_paths) - 1)))
    file = source_paths[path_index]      # Setting which file to open
    data = np.array(DF(file))   # Creates dataframe from the .dat file
    
    lab = np.zeros((len(data[:,0]),1))
    data = np.column_stack((data, lab))

    if max(data[:,0]) < 170:
        continue
    else:
        #progressBar(counter, numBursts)
        counter += 1
        
        upperTime = max(data[:,1])
        
        snArr = []              # All SN values of a burst
        dmArr = []              # All DM value of a burst
        tArr = []               # All times for the burst
        wArr = []
        labArr = []
        timeMid = np.random.uniform(1,upperTime)                # Average time of detection for a burst
        dmMid = np.random.uniform(dmRangeBot, max(data[:,0]))       # Average DM of the burst
        peakSN = np.random.uniform(snRangeBot, snRangeTop)      # Peak SN of the burst
        SNratio = np.random.uniform(0.05,0.25)
        
        noiseFraction = 0
        while (noiseFraction < 0):
            noiseFraction = np.random.normal(0.15,0.2)
            
        # Time duration of the burst
        Wms = 0
        while (Wms < 1):
            Wms = np.random.gumbel(0.13,0.12)*50     # Draws duration from this distribution
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
                wArr.append(32)
                labArr.append(1)
            
                if tempVar1 <= noiseFraction*fraction:  # Generates the the noise points
                    count += 1
                    noiseDM = np.random.uniform((dmMid + start*1.1 - stDevDM*3), (dmMid + stDevDM*3 - start*1.1))
                    noiseSN = np.random.uniform(0, peakSN*0.8)
                    noiseTime = timeMid + np.random.uniform(-timeRange,timeRange)/1000
                    
                    dmArr.append(takeClosest(DM_poss, dmMid + dmTemp + devDM))
                    snArr.append(noiseSN)
                    tArr.append(noiseTime)
                    wArr.append(32)
                    labArr.append(0)

        tArr = timeGen(tArr, dmArr, snArr)
        
        """
        upper = np.quantile(tArr, 0.9)
        lower = np.quantile(tArr, 0.1)
        print((upper - lower)*1000)
        print(Wms)
        print("\n")"""
            
        # Creates a numpy table of the burst, same format as the standard .dat files
        totArr = np.zeros([count, 5])
        totArr[:,0] = np.array(np.transpose(dmArr))
        totArr[:,1] = np.array(np.transpose(tArr))
        totArr[:,2] = np.array(np.transpose(snArr))
        totArr[:,3] = np.array(np.transpose(wArr))
        totArr[:,4] = np.array(np.transpose(labArr))
        
        finalArr = np.vstack((data,totArr))
        
        if intention == "s":
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(finalArr[:,1],finalArr[:,0], s = 5, c = finalArr[:,4], cmap = "brg", vmin = -1, alpha = 0.6)
            ax.set_xlabel("Time")
            ax.set_ylabel("DM")
            ax.set_title("Cordes function")
            plt.show()
        elif intention == "g" or intention == "t":

            finalArr.reshape(-1)
            fileEnding = "(" + str(counter) + ").dat"
            fileName = file.split("idir\\")[1].replace('.dat',fileEnding)               # Name of the class file to be created
            
            finalArr.astype(np.float32).tofile(os.getcwd() + '\\odir\\' + folderName + "\\" + fileName)

progressBar(numBursts,numBursts)         
        
counter = 0
print("\nGenerating non-candidates")

while (counter < numBursts) and (intention == "t"):
    progressBar(counter, numBursts)
    counter += 1
    
    path_index = int(round(np.random.uniform(0,len(source_paths) - 1)))
    file = source_paths[path_index]         # Setting which file to open
    data = np.array(DF(file))               # Creates dataframe from the .dat file
    
    lab = np.zeros((len(data[:,0]),1))
    data = np.column_stack((data, lab))
    
    if max(data[:,0]) < 100:
        continue
    else:
        """-------------------------------------------------------------------------"""
        
        upperTime = max(data[:,1])
        
        snArr = []              # All SN values of a burst
        dmArr = []              # All DM value of a burst
        tArr = []               # All times for the burst
        wArr = []
        labArr = []
        timeMid = np.random.uniform(1,upperTime)                # Average time of detection for a burst
        dmMid = np.random.uniform(50, max(data[:,0]))            # Average DM of the burst
        peakSN = np.random.uniform(snRangeBot, snRangeTop)      # Peak SN of the burst
        
        dmWidthTop = 0.5*dmMid
        
        # DM width of a burst
        dmWidth = 0
        while dmWidth <= 10:
            dmWidth = np.random.normal((dmWidthTop - dmWidthBot)/2,(dmWidthTop - dmWidthBot)/4)
            
        # Time duration of the burst
        Wms = 0
        while (Wms < 1):
            Wms = (np.random.gumbel(0.5,0.7)*200)     # Draws duration from this distribution
        timeRange = Wms/(2*quantFrac)       # Time range for the time distribution of the burst, centered around 0
        
        numPoints = 0       # Number of points in the burst
        while numPoints < 20:
            numPoints = int(np.random.gumbel(0.015, 0.06)*2000) # Draws the number of points from this distribution
        
        stDevSN = np.random.uniform(2,5)             # Standard deviation of the SN of the burst events
        stDevDM = dmWidth*np.random.uniform(0.1,0.4)   # Standard deviation of the DM of the burst events
        start = -dmWidth/2          # Lower end of the DM range of the burst, currently centered around 0
        step = dmWidth/numPoints    # Size of each DM step when looping through the DM range
        count = 0                   # Countiong the number of events in the burst 
    
        typeVar = np.random.uniform(0,1)
        typeVar = 0.2
        side = np.random.uniform(0,1)
        botSN = np.random.uniform(0,peakSN-1)
        horSN = np.random.uniform(9,15)
        horBotSN = np.random.uniform(9,14)
        horStDevSN = np.random.uniform(0,1)
        count = 0
        pointNum = 0
        horLinOrd = np.random.uniform(0,1)
        
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
    
                if typeVar >= 0.75:     # Linear SN shape
                    if side >= 0.5:
                        c = ((peakSN + botSN) - ((peakSN - botSN)/(dmWidth))*(2*dmMid))/2
                        k = (peakSN - botSN)/(dmWidth)
                        y = (dmMid + dmTemp)*k + c
                        snArr.append(y) # SN value of the point, including deviation
                    else:
                        c = (botSN + peakSN - ((botSN - peakSN)/(dmWidth))*(2*dmMid))/2
                        k = (botSN - peakSN)/(dmWidth)
                        y = (dmMid + dmTemp)*k + c
                        snArr.append(y) # SN value of the point, including deviation
                    dmArr.append(takeClosest(DM_poss, dmMid + dmTemp + devDM))
                        
                elif typeVar >= 0.5:    # Horizontal SN shape
                    devSN = np.random.normal(0,horStDevSN)  # Using own StDev since these tend to be narrower than normal signals
                    y = horSN + devSN
                    snArr.append(y)
                    
                elif typeVar >= 0.25:   # Random uniform SN distribution
                    y = horSN + devSN
                    snArr.append(y)
                    
                else:
                    pointNum += 1
                    if horLinOrd >= 0.5:
                        if pointNum >= numPoints/2:
                            altWidth = dmWidth/2
                            devDM = np.random.normal(0,4)
                            if side >= 0.5:
                                c = (horSN + horBotSN - ((horSN - horBotSN)/(altWidth))*(2*(dmMid + altWidth)))/2
                                k = (horSN - horBotSN)/(altWidth)
                                y1 = ((dmMid + altWidth) + dmTemp)*k + c
                                snArr.append(y1) # SN value of the point, including deviation
                            else:
                                c = (horBotSN + horSN - ((horBotSN - horSN)/(altWidth))*(2*(dmMid + altWidth)))/2
                                k = (horBotSN - horSN)/(altWidth)
                                y1 = ((dmMid + altWidth) + dmTemp)*k + c
                                snArr.append(y1) # SN value of the point, including deviation
                        else:
                            devSN2 = np.random.normal(0,horStDevSN)  # Using own StDev since these tend to be narrower than normal signals
                            y2 = horSN + devSN2
                            snArr.append(y2)
                    else:
                        if pointNum <= numPoints/2:
                            altWidth = dmWidth/2
                            devDM = np.random.normal(0,4)
                            if side >= 0.5:
                                c = (horSN + horBotSN - ((horSN - horBotSN)/(altWidth))*(2*(dmMid + altWidth)))/2
                                k = (horSN - horBotSN)/(altWidth)
                                y1 = ((dmMid + altWidth) + dmTemp)*k + c
                                snArr.append(y1) # SN value of the point, including deviation
                            else:
                                c = (horBotSN + horSN - ((horBotSN - horSN)/(altWidth))*(2*(dmMid + altWidth)))/2
                                k = (horBotSN - horSN)/(altWidth)
                                y1 = ((dmMid + altWidth) + dmTemp)*k + c
                                snArr.append(y1) # SN value of the point, including deviation
                        else:
                            devSN2 = np.random.normal(0,horStDevSN)  # Using own StDev since these tend to be narrower than normal signals
                            y2 = horSN + devSN2
                            snArr.append(y2)
                    dmArr.append(takeClosest(DM_poss, dmMid + dmTemp + devDM))
                        
                timeVar = 0.000256*round((timeMid + np.random.uniform(-timeRange,timeRange)/1000)/0.000256)     # Time of detection of the points
                tArr.append(round(timeVar,6))    # Adds the time to the time array that has been "pixelated" to match the p-band data
                wArr.append(32)
                labArr.append(0)
        
        # Creates a numpy table of the burst, same format as the standard .dat files
        totArr = np.zeros([count, 5])
        totArr[:,0] = np.array(np.transpose(dmArr))
        totArr[:,1] = np.array(np.transpose(tArr))
        totArr[:,2] = np.array(np.transpose(snArr))
        totArr[:,3] = np.array(np.transpose(wArr))
        totArr[:,4] = np.array(np.transpose(labArr))
        
        finalArr = np.vstack((data,totArr))
        
        finalArr.reshape(-1)
        fileEnding = "(" + str(counter + numBursts) + ").dat"
        fileName = file.split("idir\\")[1].replace('.dat',fileEnding)               # Name of the class file to be created
        
        finalArr.astype(np.float32).tofile(os.getcwd() + '\\odir\\' + folderName + "\\" + fileName)
    
progressBar(numBursts,numBursts) 
