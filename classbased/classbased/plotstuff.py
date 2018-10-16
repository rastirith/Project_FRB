import numpy as np
import matplotlib.pyplot as plt

Tfile = open("test.dat",'r')
data = np.fromfile(Tfile,np.float32,-1)

c = data.reshape((-1,4))
np.savetxt('txtfilename',c)

Tfile.close()

print(c)
columns = np.hsplit(c,4) #dm=0, time=1, ston=2, width=3

dm = columns[0]
time = columns[1]
ston = columns[2]
width = columns[3]

data = {'a': time,
        'c': dm,
        'd': width}

plt.scatter(time, dm, s = (width**3)/3500)

plt.xlabel('x label')
plt.ylabel('y label')

plt.title("Simple Plot")

plt.legend()

plt.show()