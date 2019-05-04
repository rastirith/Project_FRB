import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
from scipy import stats
import math

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

cuts = [1, 2/3, 1/3, 0.07]
weights = [1/7, 2/7, 4/7]

widths = []
SNs = []

for m in range(5):
    peakSN = np.random.uniform(20,200)
    tempArr = [1.0]
    tempWarr = [np.random.uniform(0.05,0.3)]
    n = int(1 + peakSN/50)
    
    band = np.random.normal(5/12, 1/18)
    tempArr.append(band)
    for l in range(n - 1):
        band *= np.random.normal(2/3, 1/6)
        tempArr.append(band)
        
    tempArr = np.array(tempArr)
    tempArr[::-1].sort()
    tempArr *= peakSN
    SNs.append(tempArr)
    widths.append(tempWarr)


SNs = np.array(SNs)

for m in range(len(SNs)):
    for k in range(len(SNs[m]) - 1):
        factor = np.random.normal(0.460, 0.066)
        w = ((SNs[m][k + 1]/SNs[m][k])**(-1/factor))*widths[m][k]
        widths[m].append(w)
    
    ratios = SNs[m]/np.amax(SNs[m])
    SNs[m] = np.array(SNs[m])
    widths[m] = np.array(widths[m])
    
    SNs[m] = SNs[m][np.nonzero(widths[m] < 40)]
    widths[m] = widths[m][np.nonzero(widths[m] < 40)]
 
    
for m in range(len(SNs)):
    fig1 = plt.figure()
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.scatter(widths[m], SNs[m])
    cordAlt = cordes(widths[m][-1])
    for k in range(int(1), int(len(SNs[m]))):
        cord = cordes(widths[m][k])
        numDMpoints = int(np.round(np.random.normal(8*cord, cord)*2))
        
        x = np.linspace(-cord, +cord, numDMpoints)
        x = np.random.choice(x, size = int(numDMpoints/2), replace = False)
        y = np.random.normal(SNs[m][k], 0.3, len(x))
        cord = np.round(cord, 3)

        x[0] = 0
        y[0] = SNs[m][0]
        
        #numDMpoints = np.round(np.random.normal(8*cord, cord)*2)
        
        xTail = np.linspace(-cordAlt-5, cordAlt+5, 2*numDMpoints)
        print(xTail)
        zeta = (6.91*10**-3)*336*(1.732**-3)*(widths[m][k]**-1)*xTail      # Zeta function in the cordes function
        zeta[zeta == 0] = 0.000001
        
        yTail = []
        for i in range(len(zeta)):
            yCord = (math.pi**(1/2))*0.5*(zeta[i]**-1)*math.erf(zeta[i])
            yCord += np.random.normal(-0.03, 0.03)
            yTail.append(yCord)
        yTail = np.array(yTail)
        
        xTail = xTail[np.nonzero(yTail < 1)]
        yTail = yTail[yTail < 1]
        
        yTail *= SNs[m][k]/0.91
        
        yTail = yTail[np.nonzero((xTail < -4) | (xTail > 4))]
        xTail = xTail[np.nonzero((xTail < -4) | (xTail > 4))]
        
        xTail = xTail[np.nonzero(yTail > SNs[m][-1] - 2)]
        yTail = yTail[yTail > SNs[m][-1] - 2]
        
        xTailVert = []
        yTailVert = []
        
        for i in range(len(yTail)):
            #upperLim = 
            num = int(np.random.uniform(0, 20))
            upLim = np.amax([yTail[i], SNs[m][-1]])
            
            temp = np.random.uniform(SNs[m][-1], yTail[i], num)
            yTailVert = np.concatenate((yTailVert, temp))
            xTailVert = np.concatenate((xTailVert, [xTail[i]]*num))
        
        
        xTailVert = xTailVert[np.nonzero(yTailVert > SNs[m][-1] - 2)]
        yTailVert = yTailVert[yTailVert > (SNs[m][-1] - 2)]
            
        cTail = np.full(len(xTail), k) 
        cVert = np.full(len(xTailVert), k)
        cScat = np.full(len(x), k)
        
        ax1 = fig1.add_subplot(111)
        ax1.scatter(x, y, s = 6, vmin = 1, vmax = len(SNs[m]), c = cScat, cmap = "gnuplot")
        ax1.scatter(xTail, yTail, s = 4, vmin = 1, vmax = len(SNs[m]), c = cTail, cmap = "gnuplot")
        ax1.scatter(xTailVert, yTailVert, s = 4, vmin = 1, vmax = len(SNs[m]), c = cVert, cmap = "gnuplot")
    ax1.set_ylim(SNs[m][-1] - 20)
    ax1.set_xlim(-cord - 5, +cord + 5)
    
    #print("\n")
#print(widths)
#print(widths)

"""
snArr = []
widthArr = np.arange(-10, 100, 0.1)
widthArr[widthArr == 0] = 0.1

freq = 1.732
bandWidth = 336
dm = 57

zeta = (6.91*10**-3)*bandWidth*(freq**-3)*(widthArr**-1)*1         # Zeta-function

for i in range(len(widthArr)):
    snArr.append(math.pi**(1/2)*0.5*(zeta[i]**-1)*math.erf(zeta[i]))           # SN value of the point, including deviation



fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(, bins = 1000)

plt.show()"""