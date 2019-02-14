import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy.stats import skew
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

estimators = []
x_vals = []
est_val = 0

dataset = pd.read_csv(os.getcwd() + "\\feature_table.csv")
    
X = dataset.iloc[:,0:4].values
Y = dataset.iloc[:,4].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0) 

iterations = (int(0.75*len(X))) - 1
#print(iterations)
for i in range(1,50):
    print("Progress: " + str(2*i) + "%")
    x_vals.append(10*i)
    clf = RandomForestClassifier(n_estimators = 10*i, max_features = 3)
    clf.fit(X_train,Y_train)
    
    y_pred = clf.predict(X_test)
    #print(y_pred)
    estimators.append(accuracy_score(Y_test, y_pred))
    
    """
    print("\n" + str(confusion_matrix(Y_test,y_pred)))  
    print(classification_report(Y_test,y_pred))  
    print("Accuracy score: " + str(accuracy_score(Y_test, y_pred)))"""

fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(x_vals, estimators, s = 15)
ax.set_xlim(left = 0)
ax.set_ylim(0,1)
print("\nFeature importances: ")
for name, importance in zip(["Shape", "Sharp", "Skewness", "Kurtosis"], clf.feature_importances_):
    print(name, "=", importance)

results = clf.predict_proba(X)
#print("Probabilities: \n" + str(results))