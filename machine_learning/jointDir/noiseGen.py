import numpy as np
from scipy import special

def funcGenerator():
    tempDM = np.linspace(0.001, 10, 10000)
    
    functions = np.array([
                special.gammaln(tempDM), np.sqrt(tempDM), np.log(tempDM),
                -special.gammaln(tempDM), -np.sqrt(tempDM), -np.log(tempDM),
                special.erf(tempDM), np.square(tempDM),
                -special.erf(tempDM), -np.square(tempDM),
                special.struve(0,tempDM), special.struve(1,tempDM), special.struve(2,tempDM),
                -special.struve(0,tempDM), -special.struve(1,tempDM), -special.struve(2,tempDM),
                [np.random.uniform(0, 5)]*len(tempDM), tempDM, -tempDM
                ])
    
    return functions

def generation(numPoints, upperDM, myFunctions):
    np.random.seed()
    
    convProb = 1/2
    

    indArr = np.arange(0, len(myFunctions), 1)
    
    numFuncs = np.random.randint(1,5)
    indices = np.random.choice(indArr, numFuncs, replace = False)

    singShapes = []
    for i in myFunctions:
        
        i -= np.amin(i)
        i += 0.001
        i /= np.amax(i)
        
        singShapes.append(i)
    
    convs = 0
    convInd = "Nah"
    
    xVals = np.array([])
    dists = np.array([])
    # Convolution probs
    for i in range(0, numFuncs):
        
        convVar = np.random.uniform(0,1)
        if (convVar < convProb):
            convs += 1
            indRange = np.delete(indArr, np.argwhere(indArr == indices[i]))
            convInd = np.random.choice(indRange, 1)[0]

            finalDist = np.convolve(singShapes[indices[i]], singShapes[convInd])
            
            xTemp = np.linspace(0.001, 1, len(finalDist))
            np.random.seed(6)
            finalDist = np.random.choice(finalDist, 5000, replace = False)
            np.random.seed(6)
            xTemp = np.random.choice(xTemp, 5000, replace = False)
            
            finalDist -= np.amin(finalDist)
            finalDist += 0.001
            finalDist /= np.amax(finalDist)
            
        else:
            finalDist = singShapes[indices[i]]
            xTemp = np.linspace(0.001, 1, len(finalDist))
        
        sdevMag = np.random.uniform(0.005, 0.1)
        devs = np.random.normal(0, sdevMag, len(finalDist))
        finalDist += devs
        
        if numFuncs > 1:
            
            np.random.seed()
            xTemp += np.random.uniform(-0.5, 0.5)
            finalDist += np.random.uniform(-0.8, 0)
            
            finalDist = finalDist[np.nonzero((xTemp <= 1))]
            xTemp = xTemp[(xTemp <= 1)]
            finalDist = finalDist[np.nonzero((xTemp >= 0))]
            xTemp = xTemp[(xTemp >= 0)]
    
            xTemp = xTemp[np.nonzero(finalDist >= 0)]
            finalDist = finalDist[finalDist >= 0]
        
        dists = np.concatenate((dists, finalDist))
        xVals = np.concatenate((xVals, xTemp))
   
    
    np.random.seed(6)
    dists = np.random.choice(dists, numPoints, replace = False)
    np.random.seed(6)
    xVals = np.random.choice(xVals, numPoints, replace = False)
    
    dists /= np.amax(dists)
    xVals /= np.amax(xVals)
    
    dists *= np.random.uniform(2, 32)
    dists += 8
    xVals *= upperDM

    return dists, xVals

"""
np.random.seed()
numArr = [
            np.random.randint(20, 30),
            np.random.randint(70,90),
            np.random.randint(130,160),
            np.random.randint(200,280),
            np.random.randint(600,800)
            ]
numPoints = np.random.choice(numArr, 1, p = [0.3,0.2,0.2,0.15,0.15])[0]     # Number of points in the burst"""


