import numpy as np 
from scipy import special
    
def randFunc(x,sdev):

    functions = [
                np.sin(x), np.cos(x), special.gammaln(x),
                np.sqrt(x), np.log(x), np.log10(x), 
                special.erf(x), np.square(x),
                special.struve(np.random.uniform(0,2),x),
                special.rgamma(x)
                ]
    
    index = np.random.randint(0, int(len(functions)))

    choice = functions[index]/max(functions[index])
    choice = (choice**2)**0.5
    choice *= np.random.uniform(0, 150)
    choice += 10
    
    for k in range(len(x)):
        devSN = np.random.normal(0,sdev)
        choice[k] += devSN
    
    return(choice)
    

