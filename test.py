import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
df = pd.read_csv('dataset/50_Startups.csv')
features = df.columns[:-1]
X = df[features]
y = df[df.columns[-1]]
for i in features:
  if pd.to_numeric(X[i], errors='coerce').notnull().all() == False:
      one_hot = pd.get_dummies(X[i])
      X = X.drop(i, axis=1)
      X = X.join(one_hot)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.8)
scaler = StandardScaler()
scaler.fit(X_train)
scaler.fit(X_test)
print(X_train)