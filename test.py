import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
df = pd.read_csv('dataset/50_Startups.csv')
features = df.columns[:-1]
X = df[features]
y = df[df.columns[-1]]
mse_train_list = []
mse_test_list = []
kf = KFold(5)

for train_index, test_index in kf.split(X):
  X_train, X_test = X.iloc[train_index], X.iloc[test_index]
  y_train, y_test = y.iloc[train_index], y.iloc[test_index]
  reg = LinearRegression()
  reg.fit(X_train, y_train)
  y_pred_train = reg.predict(X_train)
  y_pred_test = reg.predict(X_test)
  mse_train_list.append(mean_squared_error(y_train, y_pred_train))
  mse_test_list.append(mean_squared_error(y_test, y_pred_test))
print(mse_test_list)