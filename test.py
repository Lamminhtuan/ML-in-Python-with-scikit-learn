
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

df = pd.read_csv('dataset/50_Startups.csv')
features = df.columns[:-1]
features = features.tolist()
X = df[df.columns[:-1]]
y = df[df.columns[-1]]
isnumber = features.copy()
for i in features:
  if pd.to_numeric(X[i], errors='coerce').notnull().all() == False:
      isnumber.remove(i)
      one_hot = pd.get_dummies(X[i])
      X = X.drop(i, axis=1)
      X = X.join(one_hot)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
ct = ColumnTransformer([('scale', StandardScaler(), isnumber)], remainder = 'passthrough')
X_train = ct.fit_transform(X_train)
X_test = ct.fit_transform(X_test)
print(X_train)
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred_train = reg.predict(X_train)
y_pred_test = reg.predict(X_test)
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test =  mean_squared_error(y_test, y_pred_test)
print('Mean squared error on train: ', mse_train)
print('Mean squared error on test: ', mse_test)