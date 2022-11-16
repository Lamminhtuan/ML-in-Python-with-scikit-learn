import numpy as np
import cv2 
from sklearn.decomposition import PCA
a = cv2.imread('images/cat.jpg')
k = int(a.shape[0]*0.05)
r, g, b = cv2.split(a)

pca = PCA(n_components=k)

r_reduced = pca.fit_transform(r)
r_inversed = pca.inverse_transform(r_reduced).astype(np.uint8)
g_reduced = pca.fit_transform(g)
g_inversed = pca.inverse_transform(g_reduced).astype(np.uint8)
b_reduced = pca.fit_transform(b)
b_inversed = pca.inverse_transform(b_reduced).astype(np.uint8)

result = cv2.merge((r_inversed, g_inversed, b_inversed))
cv2.imshow('orgininal', a)
cv2.imshow('result', result)
cv2.waitKey(0)