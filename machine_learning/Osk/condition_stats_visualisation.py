#Condition Stats Visualisation
#Creating plots from the stats taken on search algorithm over filespace
#5MB files at end of idir excluded

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os

#opening stats file to dataframe
file = (os.getcwd()+ '\\' + 'condition_stats_sharpness')
df = pd.read_csv(file)
#print(df)

df_labels=['File_Path','DBclusters','Noiseclusters','DMlimitclusters','SNPclusters','NRFIclusters','bin1','bin2','bin3','bin4','bin5','bin6','bin7','bin8','bin9','bin10']


#cluster plot
df.plot(x='File_Path',y=['DBclusters','Noiseclusters','DMlimitclusters','SNPclusters','NRFIclusters'],kind='bar',logy = False,stacked=True,)
plt.show()
#bins plot
df.plot(x='File_Path',y=['bin1','bin2','bin3','bin4','bin5','bin6','bin7','bin8','bin9','bin10'],kind='bar',logy = False,stacked=True,)
plt.show()
#sum statistics
sums = df.sum()
print(sums)
#for i in range(1,16):
    #print(sums[i])

#percentile sum plot
fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar(x=df_labels[6:16] ,height=sums[6:16])
plt.ylabel("Frequency")
plt.xlabel("Percentage Bins")
plt.title("Signal frequency in rating percentile bin")
plt.show()