import numpy as np
from mlp import MLP


data = np.loadtxt("train4dAll.txt")
X = data[:,:-1]
y = data[:, -1].reshape(-1, 1)

mlp = MLP(hidden_layer_sizes=[8,4,2], learning_rate=0.01, epochs=100)
mlp.fit(X, y)

for sample in range(len(X)):
	print(mlp.predict(X[sample]), y[sample, 0])
