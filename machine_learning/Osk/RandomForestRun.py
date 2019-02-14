import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy.stats import skew
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import pickle

rfi = 0
least_acc = 0
good = 0
excellent = 0


dataset = pd.read_csv(os.getcwd() + "\\feature_table.csv")

X = dataset.iloc[:,0:5].values
Y = dataset.iloc[:,5].values
paths = dataset.iloc[:,6].values

clf = pickle.load(open("model.sav",'rb'))
y_pred = clf.predict(X)

print("\n" + str(confusion_matrix(Y,y_pred)))  
print(classification_report(Y,y_pred))  
print("Accuracy score: " + str(accuracy_score(Y, y_pred)))

print("\nFeature importances: ")
for name, importance in zip(["Shape", "Sharp", "Skewness", "Kurtosis", "KS-test"], clf.feature_importances_):
    print(name, "=", importance)

results = clf.predict_proba(X)

for i in range(len(results)):
    if (results[i][1] > 0.8):
       excellent += 1
    elif (results[i][1] > 0.65):
        good += 1
    elif (results[i][1] > 0.5):
        least_acc += 1
    else:
        rfi += 1

#print("Probabilities: \n" + str(results))
print("Excellent: " + str(excellent))
print("Good: " + str(good))
print("Least acceptable: " + str(least_acc))
print("RFI: " + str(rfi))
#results[0][0]