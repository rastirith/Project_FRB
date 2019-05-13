import numpy as np
from scipy import special
from matplotlib import pyplot as plt

def rescale(arr, newMin, newMax):

    oldRange = (np.amax(arr) - np.amin(arr))  
    newRange = (newMax - newMin)  
    newValue = (((arr - np.amin(arr)) * newRange) / oldRange) + newMin

    return newValue

noiseFile = np.load("noiseBases.npy")
noiseBases = noiseFile[np.arange(0, len(noiseFile), 1)]

def noiseGenerator(lineOn):

    np.random.seed()
    ind1 = np.random.randint(0, len(noiseBases))
    
    convVar = np.random.uniform(0,1)
    convProb = 16/17
    lineProb = 1/30
    
    flipVar = np.random.uniform(0,1)

    snNoiseArr = np.array(noiseBases[ind1][:,2], copy=True)
    dmNoiseArr = np.array(noiseBases[ind1][:,0], copy=True)
    
    if flipVar < 1/3:
        priorMin = np.amin(snNoiseArr)

        snNoiseArr *= -1
        snNoiseArr -= np.amin(snNoiseArr)
        snNoiseArr += priorMin
        
    elif flipVar < 2/3:
        priorMin = np.amin(dmNoiseArr)

        dmNoiseArr *= -1
        dmNoiseArr -= np.amin(dmNoiseArr)
        dmNoiseArr += priorMin
    
    if convVar < convProb:

        snNoiseArr = np.repeat(snNoiseArr, 3)
        dmNoiseArr = np.repeat(dmNoiseArr, 3)
        
        tempDM = np.linspace(0.001, 10, int(len(dmNoiseArr)/3))     # X-vals to evaluate the distributions over
        """
        functions = np.array([      # Possible distributions that noise can follow
                    special.gammaln(tempDM), np.sqrt(tempDM), np.log(tempDM),
                    -special.gammaln(tempDM), -np.sqrt(tempDM), -np.log(tempDM),
                    special.erf(tempDM), np.square(tempDM),
                    -special.erf(tempDM), -np.square(tempDM),
                    special.struve(0,tempDM), special.struve(1,tempDM), special.struve(2,tempDM),
                    tempDM, -tempDM, special.rgamma(tempDM), np.random.uniform(0, 5, len(tempDM))
                    ])"""
    
        functions = np.array([      # Possible distributions that noise can follow

                        special.struve(0,tempDM), special.struve(1,tempDM), special.struve(2,tempDM),
                        ])
        
        ind2 = np.random.randint(0, len(functions))
        
        tempY = np.array(functions[ind2], copy = True)
        tempY -= np.amin(tempY)
        tempY += 0.001
        
        dmNoiseArr -= np.amin(dmNoiseArr)
        dmNoiseArr += 0.0001
        dmNoiseArr /= np.amax(dmNoiseArr)
        
        snNoiseArr -= np.amin(snNoiseArr)
        snNoiseArr += 0.0001
        snNoiseArr /= np.amax(snNoiseArr)
        
        tempX = np.linspace(0, 1, len(tempDM))
        
        tempY_scaled = rescale(tempY, np.amin(snNoiseArr), np.amax(snNoiseArr))
        convolved = np.convolve(snNoiseArr, tempY_scaled)
    
        sdevMag = np.random.uniform(0.05*np.amax(convolved), 0.2*np.amax(convolved))             # Random magnitude of sdev for distribution
        devs = np.random.normal(0, sdevMag, len(convolved)) # Generates the deviations corresponding to above sdev
        convolved += devs   # Applies deviations to distribution
        
        convolved = rescale(convolved, np.amin(noiseBases[ind1][:,2]), np.amax(noiseBases[ind1][:,2]))
        x = np.linspace(np.amin(noiseBases[ind1][:,0]), np.amax(noiseBases[ind1][:,0]), len(convolved))
        
        np.random.seed(6)
        convolved = np.random.choice(convolved, len(tempX), replace = False)
        np.random.seed(6)
        x = np.random.choice(x, len(tempX), replace = False)

        xDev = np.random.uniform(-np.amin(x)*0.9, 50)
        x += xDev
        snDev = np.random.uniform(8 - np.amin(convolved), 5)
        convolved += snDev
        
        convolved = rescale(convolved, 0, 1)
        x = rescale(x, 0, 1)
        
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        ax2.scatter(dmNoiseArr, snNoiseArr, s = 4, c = "r", label = "Original data", alpha = 0.15)
        ax2.scatter(x, convolved, s = 6, c = "k", label = "Final data")
        ax2.plot(tempX, tempY_scaled, c = "b", label = "Conv. function", alpha = 0.3)
        ax2.set_xlabel("DM (pc " + r'$cm^-3$' + ")")
        ax2.set_ylabel("SN")
        ax2.set_title("Simulated burst")
        lgnd = plt.legend()
        lgnd.legendHandles[0]._sizes = [20]
        
        for i in range(2):
            lineVar = np.random.uniform(0,1)
            if (lineVar < lineProb and lineOn == 1):
                dmSpan = np.amax(x) - np.amin(x)
                points = np.random.randint(80,200)
                lineSN = np.random.uniform(np.amin(convolved), np.amax(convolved)*1.2)
            
                xWidth = np.random.uniform(dmSpan*0.1, dmSpan*0.25)
                linePos = np.random.uniform(np.amin(x) + xWidth*0.1, np.amax(x) - xWidth*0.1)
                
                addX = np.random.uniform(linePos, linePos + xWidth, points)
                addY = np.random.normal(lineSN, 0.005, points)
                
                x = np.concatenate((x, addX))
                convolved = np.concatenate((convolved, addY))
        
        snNoiseArr = convolved
        dmNoiseArr = x
    
    np.random.seed()
    return dmNoiseArr, snNoiseArr
noiseGenerator(0)
"""
data = noiseGenerator()   
dm = data[0]
sn = data[1]

fig1 = plt.figure()
ax = fig1.add_subplot(111)
ax.scatter(dm, sn, s = 4, c = "r")"""
