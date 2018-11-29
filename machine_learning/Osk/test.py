import numpy as np
from scipy import stats

np.random.seed(12345678)  
n1 = 200
n2 = 300 
rvs1 = stats.norm.rvs(size=n1, loc=0., scale=1)

print(rvs1)