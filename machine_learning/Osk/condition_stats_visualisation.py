#Condition Stats Visualisation
#Creating plots from the stats taken on search algorithm over filespace
#5MB files at end of idir excluded

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os

#opening stats file to dataframe
file = (os.getcwd()+ '\\' + 'condition_stats_8')
df = pd.read_csv(file)
#print(df)

#v1 (file<7)
#df_labels=['File_Path','DBclusters','Noiseclusters','DMlimitclusters','SNPclusters','NRFIclusters','bin1','bin2','bin3','bin4','bin5','bin6','bin7','bin8','bin9','bin10']
#v2 (file 7)
#df_labels=['File_Path','DBclusters','Noiseclusters','DMlimitclusters','SNclusters','NRFIclusters','FAIRclusters','GOODclusters','EXCELclusters','bin1','bin2','bin3','bin4','bin5','bin6','bin7','bin8','bin9','bin10']
#v3 (file>7)
df_labels=['File_Path','DBclusters','Noiseclusters','DMlimitclusters','SNPclusters','SNRclusters','NRFIclusters','REJECTED','FAIRclusters','GOODclusters','EXCELclusters','bin1','bin2','bin3','bin4','bin5','bin6','bin7','bin8','bin9','bin10']

#cluster plot
df.plot(x='File_Path',y=['DBclusters','Noiseclusters','DMlimitclusters','SNPclusters','SNRclusters'],kind='bar',logy = False,stacked=True,)
plt.ylabel("Number of Clusters")
plt.title("Clusters after each filtering condition")
plt.show()
#rating plot
ax = df.plot(x='File_Path',y=['REJECTED','FAIRclusters','GOODclusters','EXCELclusters'],kind='bar',logy = False,stacked=True,)#yticks=np.arrange(0,5,1)
start, end = ax.get_ylim()
ax.yaxis.set_ticks(np.arange(start, end, 1))
plt.ylabel("Number of Clusters")
plt.title("Signal Ranking For Each File")
plt.show()
#bins plot
df.plot(x='File_Path',y=['bin1','bin2','bin3','bin4','bin5','bin6','bin7','bin8','bin9','bin10'],kind='bar',logy = False,stacked=True,)
plt.ylabel("Number of Clusters")
plt.title("Signal Confidence Bin For Each File")
plt.show()

#sum statistics
sums = df.sum()
print(sums)
#for i in range(1,16):
    #print(sums[i])

#percentile sum plot
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.bar(x=df_labels[11:21] ,height=sums[11:21])
plt.ylabel("Frequency")
plt.xlabel("Percentage Bins")
plt.title("Signal frequency in rating percentile bin")
plt.show()

#Condition filtering plot
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.bar(x=df_labels[1:6] ,height=sums[1:6])
plt.ylabel("Number of Clusters")
plt.xlabel("Condition  Applied")
plt.title("Clusters after each filtering condition")
plt.show()

#Rating evaluation plot
X_values = []
H_values = []
X_values.append('Signals In')
H_values.append(sums[4]-73)
for i in range(7,11):
    X_values.append(df_labels[i])
    H_values.append(sums[i])
fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.bar(x=X_values ,height=H_values)
plt.ylabel("Number of Clusters")
plt.xlabel("Classification")
plt.title("Rating System Evaluation")
plt.show()