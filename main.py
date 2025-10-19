import numpy as np
from mlp import MLP


data = np.loadtxt("train4dAll.txt")
X = data[:,:-1]
y = data[:, -1]

mlp  = MLP()
mlp.fit(X, y)

for sample in range(len(X)):
	print(mlp.predict(X[sample]), y[sample])
