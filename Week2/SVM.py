import numpy as np
import pandas as pd
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split

col_names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 
             'Uniformity of Cell Shape', 'Marginal Adhesion', 
             'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin','Normal Nucleoli',
            'Mitoses', 'Class']
df = pd.read_csv('breast-cancer-wisconsin.data',names=col_names)

df.replace('?',-99999,inplace=True)
df.drop(['Sample code number'],1,inplace=True)

X = np.array(df.drop(['Class'],1))
y = np.array(df['Class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train_transformed = scaler.transform(X_train)

X_test_transformed = scaler.transform(X_test)

clf = svm.SVC(C=1)
clf.fit(X_train_transformed, y_train)

confidence = clf.score(X_test_transformed, y_test)
print("confidence",confidence)

example_measures = np.array([[4,2,1,1,1,2,3,2,1]])
example_measures = example_measures.reshape(len(example_measures),-1)
prediction = clf.predict(example_measures)
print("prediction", prediction)