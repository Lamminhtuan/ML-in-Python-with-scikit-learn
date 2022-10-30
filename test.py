
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
df = pd.read_csv('dataset/50_Startups.csv')
features = df.columns[:-1]
features = features.tolist()
X = df[features]
y = df[df.columns[-1]]
isnumber = features.copy()
for i in features:
  if pd.to_numeric(X[i], errors='coerce').notnull().all() == False:
      isnumber.remove(i)
      one_hot = pd.get_dummies(X[i])
      X = X.drop(i, axis=1)
      X = X.join(one_hot)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
 
# ct = ColumnTransformer([('scale', StandardScaler(), isnumber)], remainder = 'passthrough')
# scaler = ct.fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred_train = reg.predict(X_train)
y_pred_test = reg.predict(X_test)
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test =  mean_squared_error(y_true=y_test,y_pred=y_pred_test)
print(mse_test)