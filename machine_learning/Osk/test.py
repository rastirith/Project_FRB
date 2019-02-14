import numpy as np
import matplotlib.pyplot as plt

arr = np.zeros(4)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot([0,1,2],[33,4,55], color = "g", label = "True positives")
ax.plot([3,4,7],[5,22,7], color = "r", label = "False positives")
ax.set_title("BESH")
plt.legend()
plt.show()