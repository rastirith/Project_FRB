import numpy as np
from matplotlib import pyplot as plt
import math
from scipy import special

# Method to calculate and return the theoretical DM range span given a certain
# time duration/width and peak magnitude
def cordes(Wms):
    freq = 1.732
    bandWidth = 336
    cordes = []                 # Array containing all theoretical SN-values from the cordes function
    SNratio = np.random.uniform(0.87, 0.93)    # Ratio of the peak to the cutoff point in the data
                                # Need to calculate the corresponding DM range for a reduction from 
                                # 1 (peak) to this ratio (tails)
    
    x = np.linspace(-500,500,10000)   # Generates x-values for the cordes function
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
    return dm_range/2

def bandCand():
    
    finalSNarr = []
    finalDMarr = []
    finalWarr = []
    
    peakSN = np.random.uniform(20,200)
    tempArr = [1.0]
    tempWarr = [np.random.uniform(0.05,0.3)]
    
    n = int((1 + peakSN/50) + (np.random.normal(0, 1)**2)**0.5)     # Number of bands
    band = np.random.normal(5/12, 1/18)         # Band ratio wrt peak
    tempArr.append(band)
    
    for l in range(n - 1):                      # Creates remaining bands
        band *= np.random.normal(2/3, 1/6)
        tempArr.append(band)
        
    tempArr = np.array(tempArr)
    tempArr[::-1].sort()
    tempArr *= peakSN
    
    for k in range(len(tempArr) - 1):
        factor = np.random.normal(0.460, 0.066)
        w = ((tempArr[k + 1]/tempArr[k])**(-1/factor))*tempWarr[k]
        tempWarr.append(w)
    
    tempArr = np.array(tempArr)
    tempWarr = np.array(tempWarr)
    
    tempArr = tempArr[np.nonzero(tempWarr < 40)]
    tempWarr = tempWarr[np.nonzero(tempWarr < 40)]
     
    cordAlt = cordes(tempWarr[-1])
    for k in range(int(1), int(len(tempArr))):
        cord = cordes(tempWarr[k])
        numDMpoints = int(np.round(np.random.normal(8*cord, cord)*2))
        
        x = np.linspace(-cord, +cord, numDMpoints)
        x = np.random.choice(x, size = int(numDMpoints/2), replace = False)
        y = np.random.normal(tempArr[k], 0.3, len(x))
        w = np.random.normal(tempWarr[k], 0.01, len(x))
        cord = np.round(cord, 3)
    
        x[0] = 0
        if k == 1:
            y[0] = tempArr[0]
        
        tWidth = 0.4 + (np.log2(tempWarr[k])/10)
        
        xTail = np.linspace(-cordAlt-5, cordAlt+5, 1*numDMpoints)
        zeta = (6.91*10**-3)*336*(1.732**-3)*(tWidth**-1)*xTail      # Zeta function in the cordes function
        zeta[zeta == 0] = 0.000001
            
        yDeviations = np.random.normal(0,0.02, len(zeta))
        yTail = (math.pi**(1/2))*0.5*(zeta**-1)*special.erf(zeta) + yDeviations
        wTail = np.random.normal(tempWarr[k], 0.01, len(zeta))

    
        yTail = np.array(yTail)
        wTail = np.array(wTail)
        
        capRatio = np.random.uniform(tempArr[-1]/tempArr[0] + 0.08, 0.24)
        
        xTail = xTail[np.nonzero(yTail < capRatio)]
        wTail = wTail[np.nonzero(yTail < capRatio)]
        yTail = yTail[yTail < capRatio]
        
        yTail *= tempArr[0]
        
        xTail = xTail[np.nonzero(yTail > tempArr[-1])]
        wTail = wTail[np.nonzero(yTail > tempArr[-1])]
        yTail = yTail[yTail > tempArr[-1]]
        
        xTailVert = []
        yTailVert = []
        wTailVert = []
        
        for i in range(len(yTail)):
            randVar = np.random.uniform(0,1)
            if randVar > (i/(2.5*len(yTail))) + 0.2 - k*0.1:
                num = np.random.randint(0, 5)
                temp = np.random.uniform(tempArr[-1], yTail[i], num)
                yTailVert = np.concatenate((yTailVert, temp))
                xTailVert = np.concatenate((xTailVert, [xTail[i]]*num))
                wTailVert = np.concatenate((wTailVert, np.random.normal(tempWarr[k], 0.01, num)))
        
        """
        iVals = np.arange(0, len(yTail), 1)
        ifArr = (iVals/(2.5*len(yTail))) + 0.2 - k*0.1
        randVars = np.random.uniform(0, 1, len(yTail))
        trueInds = iVals[np.nonzero(randVars > ifArr)]
        nums = np.random.randint(0, 5, len(trueInds))"""
        
        
        if len(yTailVert) > 0:
            xTailVert = xTailVert[np.nonzero(yTailVert > (tempArr[-1] - 2))]
            wTailVert = wTailVert[np.nonzero(yTailVert > (tempArr[-1] - 2))]
            yTailVert = yTailVert[yTailVert > (tempArr[-1] - 2)]
            
        finalSNarr = np.concatenate((finalSNarr, yTailVert, yTail, y))
        finalDMarr = np.concatenate((finalDMarr, xTailVert, xTail, x))
        finalWarr = np.concatenate((finalWarr, wTailVert, wTail, w))
    
    addRight = np.random.randint(1,4)
    for q in range(addRight):
        
        step = np.random.uniform(0.05, 0.15)
        tWidth += step
        
        xTail = np.linspace(-cordAlt-5, cordAlt+5, 1*numDMpoints)
        zeta = (6.91*10**-3)*336*(1.732**-3)*(tWidth**-1)*xTail      # Zeta function in the cordes function
        zeta[zeta == 0] = 0.000001
        
        yDeviations = np.random.normal(0,0.02, len(zeta))
        yTail = (math.pi**(1/2))*0.5*(zeta**-1)*special.erf(zeta) + yDeviations
        wTail = np.random.normal(tempWarr[-1], 0.01, len(zeta))
        
        capRatio = np.random.uniform(tempArr[-1]/tempArr[0] + 0.08, 0.24)
        
        xTail = xTail[np.nonzero(yTail < capRatio)]
        wTail = wTail[np.nonzero(yTail < capRatio)]
        yTail = yTail[yTail < capRatio]
        
        yTail *= tempArr[0]
        
        xTail = xTail[np.nonzero(yTail > tempArr[-1])]
        wTail = wTail[np.nonzero(yTail > tempArr[-1])]
        yTail = yTail[yTail > tempArr[-1]]
        
        xTailVert = []
        yTailVert = []
        wTailVert = []
        
        for i in range(len(yTail)):
            randVar = np.random.uniform(0,1)
            if randVar > (i/(2.5*len(yTail))) + 0.2 - k*0.1:
                num = int(np.random.uniform(0, 5))
    
                temp = np.random.uniform(tempArr[-1], yTail[i], num)
                yTailVert = np.concatenate((yTailVert, temp))
                xTailVert = np.concatenate((xTailVert, [xTail[i]]*num))
                wTailVert = np.concatenate((wTailVert, np.random.normal(tempWarr[-1], 0.01, num)))
        
        if len(yTailVert) > 0:
            xTailVert = xTailVert[np.nonzero(yTailVert > (tempArr[-1] - 2))]
            wTailVert = wTailVert[np.nonzero(yTailVert > (tempArr[-1] - 2))]
            yTailVert = yTailVert[yTailVert > (tempArr[-1] - 2)] 
            
        finalSNarr = np.concatenate((finalSNarr, yTailVert, yTail))
        finalDMarr = np.concatenate((finalDMarr, xTailVert, xTail))
        finalWarr = np.concatenate((finalWarr, wTailVert, wTail))
            
        
    return finalDMarr, finalSNarr, finalWarr

bandCand()