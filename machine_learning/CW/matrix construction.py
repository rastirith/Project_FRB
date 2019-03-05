#Matix construction for CNN
import numpy as np
import pandas as pd
import glob, os

###NEED TO HANDLE SEPERATE CLUSTERS SO ADD COLUMN TO DATA FILES WITH LABEL
###see sourcepaths[16]&8



#Import data
#array of file locations and chosing the file to inspect with path
source_paths = []

#file number in source_paths to open
x = 0

#filling source_paths from the idir
for file in glob.glob(os.getcwd() + '\candidates\\' + "*.csv"):
    source_paths.append(file)
    
df = pd.read_csv(source_paths[x],index_col=0,)
"""print(df)"""
df = df.drop(columns=["Class"])
df = df.round({"DM": 3}) #have to round fp error out of dm
print(df)

#import dedispersion plan
df_ddp = pd.read_csv("dd_plan.txt")
"""print(df_ddp)"""
#setup array for step limits
dd_DM = np.array(df_ddp["DM_stop"])
dd_step = np.array(df_ddp["DM_step"])

#set up arrays from data
###incase they becom useful
DM = np.array(df["DM"])
TIME = np.array(df["Time"])
SN = np.array(df["S/N"])

#find ranges of time and dm
#already sorted by dm so use difference between first and last row
"""
df_dmrange = pd.concat([df.head(1),df.tail(1)])
dmrange = np.array(df_dmrange["DM"])
print(dmrange)
"""
dmrange = np.array([DM[0],DM[-1]])

###Starting the new method for comparisons
DM_start = df_ddp["DM_start"]
DM_stop = df_ddp["DM_stop"]
DM_step = df_ddp["DM_step"]

DM_poss = [0.0]

for i in range(len(DM_stop)):
    DM_range=DM_stop[i]-DM_start[i]
    num=round(DM_range/DM_step[i])
       
    for j in range(int(num)):
        DM_poss.append(round(DM_poss[-1]+DM_step[i],3))
        
"""print(DM_poss)"""
 
DF = pd.DataFrame(DM_poss,columns=["DM"])
"""print(DF.head)"""

#getting dm range, n dimension value
a=DF.loc[DF["DM"] == dmrange[0]].index.item()
b=DF.loc[DF["DM"] == dmrange[1]].index.item()

n=b-a

print("n=", n)
#time range, m dimension 
###same as before
t_step = 256*(pow(10,-6))

d= round(np.amin(TIME)/t_step)
c= round(np.amax(TIME)/t_step)

print("d=",d)
print("c=",c)

m = c-d
m=int(m)

print("m=",m)


#matrix construction
#zero matrix of required dimension
###+1 to handle the fact you need to count the first value istelf

zero = np.zeros((n+1,m+1))
v_array=[]
u_array=[]
#find position of data point
for i in range(len(DM)):
    #dm axis location
    u = DF.loc[DF["DM"] == DM[i]].index.item()
    u = u-a
    u_array.append(u)
    #time axis location
    v = round(TIME[i]/t_step)
    v = v-d
    v = int(v)
    v_array.append(v)
    #filling position
    zero[u][v]=SN[i]

print(np.unique(zero))
"""
print("len=", len(DM))
print(len(zero))
print(zero)
"""
np.savetxt("matrix.txt", zero)   


###need to downsample in time by some bin amount
#pixels wanted in time axis
r_pixels=100

#new time downsampled matrix of wanted size
zero2=np.zeros((n+1,r_pixels))

for i in range(len(zero[:,:])):
    ##iterator over num of rows
    #downsample into r_pixels in time axis
    dummy=np.array_split(zero[i,:],r_pixels)
    dummy2=[]
    for j in range(len(dummy)):
        #take max from each pixel group to give value to new pixel
        dummy2.append(max(dummy[j]))
    #fill new matrix by rows
    zero2[i]=dummy2


###code from old method left incase of need
"""
#tracking if step upper limit has been set
set_1 = False
set_2 = False
for i in range(len(dd_DM)):
    if dmrange[0] < dd_DM[i] and set_1==False:
        lim_1_val = dd_DM[i]
        lim_1_ind = i
        set_1 = True
   
    if dmrange[1] < dd_DM[i] and set_2==False:
        lim_2_val = dd_DM[i]
        lim_2_ind = i
        set_2 = True

#seting a DM dimension number for matrix
###need to find the number of steps separating smallest and largest dm value 
span=np.arange(lim_1_ind,lim_2_ind+1,step=1)

n=0 #number of dm steps in candidate DM range
dd_DM = np.insert(dd_DM,0,0) #adjusting for zero
step_num = []
for i in span:
    if dmrange[1]<=dd_DM[i+1] and dmrange[0]>=dd_DM[i]:
        a=(dmrange[1]-dmrange[0])/dd_step[i]
        n+=round(a,3)
        step_num.append(round(a,3))
        break
    if dmrange[1]>=dd_DM[i+1] and dmrange[0]>=dd_DM[i]:
        a=(dd_DM[i+1]-dmrange[0])/dd_step[i]
        n+=round(a,3)
        step_num.append(round(a,3))
    elif dmrange[1]>=dd_DM[i+1]:
        a=round((dd_DM[i+1]-dd_DM[i]),3)/dd_step[i]
        n+=round(a,3)
        step_num.append(round(a,3))
    else:
        a=round((dmrange[1]-dd_DM[i]),3)/dd_step[i]
        n+=round(a,3)    
        step_num.append(round(a,3))


#Time dimension
t_step = 256*(pow(10,-6))
t_range = np.amax(TIME) - np.amin(TIME)

m = round(t_range/t_step)


#construct matrix of zeroes 
#need ints
n=int(n)
m=int(m)

print(n)   
print(step_num)
print(m)

zero = np.zeros((n,m))

print(span)
print(step_num)
#fill matrix
###need trick to reference steps from begining of file
for i in range(len(DM)):
    #n position
    ###compare step_num with span and dd_step to find difference from n=0
    
    #m position
    m_pos = int((TIME[i]-TIME[0])/t_step)


print("\nBREAK \n\n")
"""
