from sklearn.preprocessing import MinMaxScaler
data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
print(data)
scaler = MinMaxScaler()
print(scaler.fit(data))

print(scaler.data_max_)

print(scaler.transform(data))

print(scaler.transform([[2, 2]]))
