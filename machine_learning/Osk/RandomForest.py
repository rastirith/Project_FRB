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

dataset = pd.read_csv(os.getcwd() + "\\feature_table.csv")
    
X = dataset.iloc[:,0:5].values
Y = dataset.iloc[:,5].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0) 


clf = RandomForestClassifier(n_estimators = 30, max_features = 2)
clf.fit(X_train,Y_train)
        
y_pred = clf.predict(X_test)

print("\n" + str(confusion_matrix(Y_test,y_pred)))  
print(classification_report(Y_test,y_pred))  
print("Accuracy score: " + str(accuracy_score(Y_test, y_pred)))

print("\nFeature importances: ")
for name, importance in zip(["Shape", "Sharp", "Skewness", "Kurtosis", "KS-test"], clf.feature_importances_):
    print(name, "=", importance)

results = clf.predict_proba(X)

filename = "model.sav"
pickle.dump(clf,open(filename,'wb'))



#print("Probabilities: \n" + str(results))