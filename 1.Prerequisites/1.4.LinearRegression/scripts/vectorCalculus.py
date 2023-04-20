import numpy as np

num_features = 4
batch_size = 2


X = np.arange(8, dtype=np.float32).reshape(batch_size, num_features)
Y_target = np.arange(batch_size).reshape(batch_size, -1)

W = np.arange(num_features).reshape(-1, num_features)
b = 0.5

Y_predict = X @ W.T + b

cost = 0.5*np.sum(Y_predict-Y_target)**2

# dL_dC = 