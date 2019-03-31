import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
from scipy import stats



x1 = [[0], [1], [2], [3]]
y1 = [[0], [1], [2], [3]]
colors = [[1,1,1,1], [2,2,2,2], [3,3,3,3], [4,4,4,4]]
colors1 = [[1], [2], [3], [4]]

fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(len(colors)):
    ax.scatter(x1[i], y1[i], cmap = "brg", vmin = -1, vmax = 5, c = colors1[i])
    
ax.set_xlim(0,5)
plt.show()