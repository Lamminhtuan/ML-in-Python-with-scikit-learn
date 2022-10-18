from sklearn.preprocessing import Normalizer
x = [[-4,1,2,2],[1,3,9,3]]
transformer = Normalizer(norm = 'l1').fit(x)
print(transformer.transform(x))