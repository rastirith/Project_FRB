import numpy as np
from matplotlib import pyplot as plt
import math
from scipy import special
import scipy.stats as stats

def cordes(Wms):
    """ Calculate the dm range spanned by band at random value between
        0.87 and 0.93 in the cordes function.

    Keyword arguments:
    Wms -- the width (ms) property of signal under consideration
    """
    freq = 1.732        # Centre frequency of survey
    bandWidth = 336     # Bandwidth of survey
    SNratio = np.random.uniform(0.87, 0.93)    # Ratio of the peak to the cutoff point in the data
    
    x = np.linspace(-500,500,10000)                             # Generates x-values for the cordes function
    zeta = (6.91*10**-3)*bandWidth*(freq**-3)*(Wms**-1)*x       # Zeta function in the cordes function
    y = (math.pi**(1/2))*0.5*(zeta**-1)*special.erf(zeta)       # Values of cordes function, ranges between 0 and 1
    
    dm = x[np.nonzero(y > SNratio)]     # Array containing all dm values corresponding to y-values > SNratio
    dm_range = dm[-1] - dm[0]           # Theoretical allowed DM range for the current candidate
    
    return dm_range/2

def bandCand():
    """Simulates DM, SN, and width data for a candidate signal"""
    
    finalSNarr = []     # Simulated SN data
    finalDMarr = []     # Simulated DM datq
    finalWarr = []      # Simulated width (ms) data
    
    peakSN = 0          # SN value of the peak
    while peakSN < 20:  # Peak SN is always higher than 20
        peakSN = (np.random.gumbel(1, 1)**0.85)*55
        
    tempArr = [1.0]     # Cordes function values corresponding to the SN values of "bands", peak SN has value 1.0.
    tempWarr = [np.random.uniform(0.05,0.3)]    # Widths of the "bands", peak has small width (0.05ms to 0.3ms)
    
    n = int((1 + peakSN/50) + (np.random.normal(0, 1)**2)**0.5)     # Number of bands, derived empirically from Crab data
    band = np.random.normal(5/12, 1/18)                             # Cordes value of band
    tempArr.append(band)
    
    for l in range(n - 1):                      # Creates remaining bands underneath the second
        band *= np.random.normal(2/3, 1/6)
        tempArr.append(band)
        
    tempArr = np.array(tempArr)
    tempArr[::-1].sort()            # Sorts the array in descending order
    tempArr *= peakSN               # Assigns actual SN values 
    
    for k in range(len(tempArr) - 1):   # Assigns width data for all bands
        factor = np.random.normal(0.460, 0.066)                     # Exponential decay factor, dervied empirically from Crab data
        w = ((tempArr[k + 1]/tempArr[k])**(-1/factor))*tempWarr[k]  # Uses ratios of SN to calculate corresponding width data
        tempWarr.append(w)
    
    tempArr = np.array(tempArr)
    tempWarr = np.array(tempWarr)
    
    tempArr = tempArr[np.nonzero(tempWarr < 40)]        # Bands considered should have width less than 40
    tempWarr = tempWarr[np.nonzero(tempWarr < 40)]
    
    if len(tempArr) == 1:       # Exception case where all bands but the peak has been sorted away
        np.append(tempArr, np.random.uniform(10,12))        # Creates bottom band of low DM (10 to 12)
        factor = np.random.normal(0.460, 0.066)             
        w = ((tempArr[-1]/tempArr[0])**(-1/factor))*tempWarr[0]
        np.append(tempWarr, w)
     
    cordAlt = cordes(tempWarr[-1])      # Calculates DM range spanned by bottom band, used to guide tail points data range
    
    """Loops through each band (index k) and generates all data for bands and tails"""
    for k in range(int(1), int(len(tempArr))):
        cord = cordes(tempWarr[k])      # DM range spanned by current band
        numDMpoints = int(np.round(np.random.normal(8*cord, cord)*2))   # Number of points in band
        
        bandDMs = np.linspace(-cord, +cord, numDMpoints)    # Generates a range of DM data twice as large as needed
        bandDMs = np.random.choice(bandDMs, size = int(numDMpoints/2), replace = False)     # Randomly reduces the range to half (simulates randomness of points)
        bandSNs = np.random.normal(tempArr[k], 0.3, len(bandDMs))   # All SN data in the band
        bandWs = np.random.normal(tempWarr[k], 0.01, len(bandDMs))  # All width data in the band

        if k == 1:
            np.append(bandDMs, 0)
            np.append(bandSNs, tempArr[0])
        
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
        
        if len(yTailVert) > 0:
            xTailVert = xTailVert[np.nonzero(yTailVert > (tempArr[-1] - 2))]
            wTailVert = wTailVert[np.nonzero(yTailVert > (tempArr[-1] - 2))]
            yTailVert = yTailVert[yTailVert > (tempArr[-1] - 2)]
            
        finalSNarr = np.concatenate((finalSNarr, yTailVert, yTail, bandSNs))
        finalDMarr = np.concatenate((finalDMarr, xTailVert, xTail, bandDMs))
        finalWarr = np.concatenate((finalWarr, wTailVert, wTail, bandWs))
    
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
            
    noiseFraction = 1/4
    
    noiseNum = int((noiseFraction*len(finalDMarr))/(1 - noiseFraction))
    noiseDM = np.random.uniform(np.amin(finalDMarr), np.amax(finalDMarr), noiseNum)
    noiseSN = np.random.normal(8, np.amax([2, np.amax(finalSNarr)/20]), len(noiseDM))
    
    noiseDM = noiseDM[np.nonzero(noiseSN > 8)]
    noiseSN = noiseSN[noiseSN > 8]
    noiseW = np.full((len(noiseDM)), 32)
    
    finalSNarr = np.concatenate((finalSNarr, noiseSN))
    finalDMarr = np.concatenate((finalDMarr, noiseDM))
    finalWarr = np.concatenate((finalWarr, noiseW))
    
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(finalDMarr,finalSNarr, s = 4)
    ax.set_title(np.amax(finalSNarr/30))"""
    
    return finalDMarr, finalSNarr, finalWarr

bandCand()