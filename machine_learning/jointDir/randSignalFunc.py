import numpy as np 
from scipy import special
from matplotlib import pyplot as plt
    
def randFunc(x,sdev):

    tempDM = x/np.amax(x)
    functions = [
                special.gammaln(tempDM),
                np.sqrt(tempDM), np.log(tempDM), np.log10(tempDM), 
                special.erf(tempDM), np.square(tempDM),
                special.struve(np.random.uniform(0,2),tempDM)
                #special.rgamma(x)
                ]

    randVar = np.random.uniform(0,1)
    combProb = 1/2
    
    if randVar < combProb:
        indices = np.random.randint(0, len(functions), 2)
        distA = functions[indices[0]]
        distB = functions[indices[1]]
        choice = np.concatenate((distA, distB))
        choice = np.random.choice(choice, size = len(tempDM), replace = False)
    else:
        indices = np.random.randint(0, int(len(functions)))
        choice = functions[indices]

    choice /= np.amax(choice)
    choice = (choice**2)**0.5
    choice *= np.random.uniform(2, 32)
    choice += 8
    
    for k in range(len(tempDM)):
        devSN = np.random.normal(0,sdev)
        choice[k] += devSN
        
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.scatter(x, choice)
    ax2.set_title(indices)
    
    return(choice)
    

