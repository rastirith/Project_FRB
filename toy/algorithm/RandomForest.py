import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import pickle

dataset = pd.read_csv(os.getcwd() + "\\candGeneration\\odir\\algTrain1\\feature_table.csv")
    
X = dataset.iloc[:,0:4].values
Y = dataset.iloc[:,4].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0) 


clf = RandomForestClassifier(n_estimators = 100, max_features = 'auto')
clf.fit(X_train,Y_train)
        
y_pred = clf.predict(X_test)

print("\n" + str(confusion_matrix(Y_test,y_pred)))  
print(classification_report(Y_test,y_pred))  
print("Accuracy score: " + str(accuracy_score(Y_test, y_pred)))

print("\nFeature importances: ")
for name, importance in zip(["Shape", "Skewness", "Kurtosis", "KS-test"], clf.feature_importances_):
    print(name, "=", importance)

results = clf.predict_proba(X)
"""
filename = "model.sav"
pickle.dump(clf,open(filename,'wb'))



#print("Probabilities: \n" + str(results))"""