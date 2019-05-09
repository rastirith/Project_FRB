import numpy as np
from scipy import special
from matplotlib import pyplot as plt


tempDM = np.linspace(0.001, 10, 10000)      # X-vals to evaluate the distributions over
    
functions = np.array([      # Possible distributions that noise can follow
            special.gammaln(tempDM), np.sqrt(tempDM), np.log(tempDM),
            -special.gammaln(tempDM), -np.sqrt(tempDM), -np.log(tempDM),
            special.erf(tempDM), np.square(tempDM),
            -special.erf(tempDM), -np.square(tempDM),
            special.struve(0,tempDM), special.struve(1,tempDM), special.struve(2,tempDM),
            -special.struve(0,tempDM), -special.struve(1,tempDM), -special.struve(2,tempDM),
            [np.random.uniform(0, 5)]*len(tempDM), tempDM, -tempDM, special.rgamma(tempDM)
            ])


def generation(numPoints, upperDM):
    """Randomly pick shapes for noise to follow. With probability of multiple, different,
    distributions (as well as their inverted counterparts) to be convolved with each other and
    form unpredictable shapes.

    Keyword arguments:
    numPoints -- the number of points that the noise/fake signal will have
    upperDM -- the highest DM for the data (w.r.t. 0), to be multiplied by the normalised plot in the end
    """
    
    np.random.seed()    # As random seeds are used at some points this just ensure it is reset every time
    
    convProb = 3/4      # Probability of convolution of a signal taking place

    indArr = np.arange(0, len(functions), 1)    # Array of all possible indices that can be chosen from 'functions' array
    
    numFuncs = np.random.randint(1,6)       # Number of distributions to be included in this single data set
    indices = np.random.choice(indArr, numFuncs, replace = True)   # Chooses, at random, distributions for the dataset

    singShapes = []         # Shifted to positive and normalised distributions
    for i in functions:
        i -= np.amin(i)     # Shifts distribution to begin at y-value (SN) 0
        i += 0.001          # Shifts it slightly above 0 to avoid complications with some distributions
        i /= np.amax(i)     # Normalises the distribution
        
        singShapes.append(i)
    
    xVals = np.array([])    # Final x-values (DM) of the dataset
    dists = np.array([])    # Final y-values (SN) of the dataset
    
    # For each distributions, gives it prob. to be convolved. Adds sdev in SN space to each distribution as well
    for i in range(0, numFuncs):
        
        convVar = np.random.uniform(0,1)    # Random variable determining whether or not dist. is convolved
        if (convVar < convProb):            # Goes here to be convolved
            indRange = np.delete(indArr, np.argwhere(indArr == indices[i])) # Range of indices of functions dist. can be convolved with
                                                                            # Does not include the index of the function under consideration
            convInd = np.random.choice(indRange, 1)[0]  # Picks one of the remaining indices at random from the array

            finalDist = np.convolve(singShapes[indices[i]], singShapes[convInd])    # Returns array of convolved distribution
            
            xTemp = np.linspace(0.001, 1, len(finalDist))   # Generates x-vals corresponding to the new distribution
            np.random.seed(6)
            finalDist = np.random.choice(finalDist, 10000, replace = False)  # Picks a number of the points at random (reduces array)
            np.random.seed(6)
            xTemp = np.random.choice(xTemp, 10000, replace = False)          # Picks the corresponding x-vals with help of the random seed
            
            finalDist -= np.amin(finalDist)     # Shifts the new distribution to start at 0
            finalDist += 0.001                  # Shifts it slightly above 0
            finalDist /= np.amax(finalDist)     # Normalises the new distribution
            
        else:   # If no convolution, go here and simply add the distribution to the final array as is
            finalDist = singShapes[indices[i]]
            xTemp = np.linspace(0.001, 1, len(finalDist))
        
        sdevMag = np.random.uniform(0.005, 0.1)             # Random magnitude of sdev for distribution
        devs = np.random.normal(0, sdevMag, len(finalDist)) # Generates the deviations corresponding to above sdev
        finalDist += devs   # Applies deviations to distribution
        
        if numFuncs > 1:    # If two or more distribution in one dataset, go here
                            # Gives individual distributions a random position in relation to each other
            np.random.seed()
            xTemp += np.random.uniform(-0.5, 0.5)       # Random shift of DM data
            finalDist += np.random.uniform(-0.8, 0)     # Random shift of SN data
            
            # Only keeps values between 0 and 1 in both DM and SN space
            finalDist = finalDist[np.nonzero((xTemp <= 1))] 
            xTemp = xTemp[(xTemp <= 1)]
            finalDist = finalDist[np.nonzero((xTemp >= 0))]
            xTemp = xTemp[(xTemp >= 0)]
    
            xTemp = xTemp[np.nonzero(finalDist >= 0)]
            finalDist = finalDist[finalDist >= 0]
        
        dists = np.concatenate((dists, finalDist))  # Adds distribution to final total array
        xVals = np.concatenate((xVals, xTemp))
   
    # Picks the correct number of points, as given by numPoints, at random from the arrays
    np.random.seed(6)
    dists = np.random.choice(dists, numPoints, replace = False) 
    np.random.seed(6)
    xVals = np.random.choice(xVals, numPoints, replace = False)
    
    dists /= np.amax(dists) # Normalises final data one last time
    xVals /= np.amax(xVals)
    
    dists *= np.random.uniform(2, 32)   # Absolute peak SN values of data (to be shifted another 8)
    dists += 8                          # Shifted up 8 SN (corresponding to cutoff in real data)
    xVals *= upperDM                    # Absolute values of DM given
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xVals, dists, s = 4)

    return dists, xVals 


np.random.seed()
numArr = [
            np.random.randint(20,30),
            np.random.randint(70,90),
            np.random.randint(200,280),
            np.random.randint(450,700)
            ]
numPoints = np.random.choice(numArr, 1, p = [0.45,0.15,0.15,0.25])[0]     # Number of points in the burst

generation(numPoints, 40)
