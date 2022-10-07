import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
def one_hot(x):
    classes, index = np.unique(x, return_inverse=True)

    one_hot_vectors = np.zeros((x.shape[0], len(classes)))
    for i, cls in enumerate(index):
        one_hot_vectors[i, cls] = 1
    return one_hot_vectors
names = np.array(['a','b','c','a','c'])
one_hot_vectors = one_hot(names)
print(one_hot_vectors)
data = pd.read_csv('.\dataset\Position_Salaries.csv')
data = data.to_numpy()
X = data[:,:-1]
Y = data[:, -1]
X_onehot = one_hot(X[:,0])
#print(X_onehot.shape)
transformed_X = np.concatenate([X_onehot, X[:,1:]], axis = 1)
#print('Transformed X', transformed_X.shape)
X_train, X_test, Y_train, Y_test = train_test_split(transformed_X, Y, test_size=0.3)


model1 = DecisionTreeRegressor(max_depth=2, random_state=0)
model2 = RandomForestRegressor(max_depth=2, random_state=0)
model1.fit(X_train, Y_train)
model2.fit(X_train, Y_train)
Y_pred1 = model1.predict(X_test)
Y_pred2 = model2.predict(X_test)
l2 = ((Y_pred1 - Y_test) ** 2).sum()
l2 = (l2 ** 0.5) / Y_test.shape[0]
print('L2 loss of model1: ', l2)
l2 = ((Y_pred2 - Y_test) ** 2).sum()
l2 = (l2 ** 0.5) / Y_test.shape[0]
print('L2 loss of model2: ', l2)
#l1 = (Y_pred - Y_test) 
#l1 = np.absolute(l1).sum()
#l1 = l1 /Y_test.shape[0]
#print('L1 loss: ', l1)
#plt.scatter(X, Y)
#plt.show()