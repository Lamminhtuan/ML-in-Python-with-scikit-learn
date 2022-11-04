import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
df = pd.read_csv('datasets/Social_Network_Ads.csv')
X = df.iloc[:, :-1]
y = df.iloc[:,-1]
reg = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f1_score(y_test, y_pred))