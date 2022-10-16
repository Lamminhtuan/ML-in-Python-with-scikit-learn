from sklearn.preprocessing import StandardScaler
data = [[1,2], [3,4]]
scaler = StandardScaler()
scaler.fit(data)
print(scaler.mean())
