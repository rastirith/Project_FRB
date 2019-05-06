import matplotlib.pyplot as plt
import numpy as np 
import random
import math
import scipy
    
def randFunc(x):

    functions = [
                np.sin(x), np.cos(x), scipy.special.gammaln(x),
                np.sqrt(x), np.log(x), np.log10(x), 
                scipy.special.erf(x), np.square(x),
                scipy.special.struve(np.random.uniform(1,4),x),
                scipy.special.rgamma(x)
                ]
    
    index = np.random.randint(0, int(len(functions)))

    choice = functions[index]/max(functions[index])
    choice = (choice**2)**0.5
    choice *= np.random.uniform(0, 150)
    choice += 8
    
    print(index)
    
    return(choice)
    

