from matplotlib import pyplot as plt
import numpy as np

x = [1, 2, 3]
y = np.log10([1000000, 116060, 6911])
z1 = [5000, 4831, 4775]
z2 = [995000, 109093, 2192]
col = [0, 1]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar(x, z1)
ax.bar(x, z2, bottom = z1)
"""
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off"""


x_labels = ["Before CNN", "After CNN"]
plt.xticks(x, x_labels)


