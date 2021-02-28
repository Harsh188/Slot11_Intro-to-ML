import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
%matplotlib inline

df = pd.read_csv('winequality-red.csv',sep=';')
print(df.head())

X = df[["alcohol","volatile acidity","sulphates","total sulfur dioxide"]]
print(X.head())

y = df[["quality"]]
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(X,y)

regression = LinearRegression()
regression.fit(X_train, y_train)
y_predicted = regression.predict(X_test)

print('R-score is: ',regression.score(X_test,y_test))