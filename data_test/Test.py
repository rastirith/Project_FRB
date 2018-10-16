import numpy as np;
import astropy as ap;
import matplotlib as plot;
import io;

from astropy.io import ascii;

#data = ascii.read("data/57886_43026_J0000+0000_000000.dat");

#print(data)

Tfile = open('Data/57886_43026_J0000+0000_000000.dat','r');
data = np.fromfile(Tfile,np.float32,-1);
#print(len(data));
#print(len(data)/4);
#print(data)
#npdata = np.array(data);
#print(len(npdata));
#print(npdata[1])
c = data.reshape((-1,4))
#print(len(c));
#for i in range(3770):
    #print(c[i])
#np.savetxt('2dummmy.txt',npdata)
np.savetxt('dummy.txt',c)
#for i in range (len(data)):
   #npdata = np.array((3770,4),data[i]);
#for i in range(len(npdata)):    
#print(npdata);

#for i in range(len(data)):
   # print(data[i]);
Tfile.close();
print(c.dtype);
print(c[0][0]);
x = np.split(c[0],4);
print(x[0])
 
#for i in range(10):