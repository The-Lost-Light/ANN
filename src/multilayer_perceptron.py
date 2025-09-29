import numpy as np
import matplotlib.pyplot as plt
import lib

def mlp_predict(weights, x):
    y = x
    for w in weights:
        y = lib.sigmoid(w @ np.array([-1, *y]))
    return lib.classify(y)

def plot_decision_boundary_2d_mlp(weights, data_filtered, title):
    x_min, x_max = np.min(data_filtered[:,0]), np.max(data_filtered[:,0])
    y_min, y_max = np.min(data_filtered[:,1]), np.max(data_filtered[:,1])
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = np.array([mlp_predict(weights, p) for p in grid]).reshape(xx.shape)

    plt.contourf(xx, yy, zz, alpha=0.3, cmap="Paired")
    plt.scatter(data_filtered[:,0][data_filtered[:,-1]==1], data_filtered[:,1][data_filtered[:,-1]==1], c="red", label="Class 1")
    plt.scatter(data_filtered[:,0][data_filtered[:,-1]==0], data_filtered[:,1][data_filtered[:,-1]==0], c="blue", label="Class 0")
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()

def plot_decision_boundary_3d_mlp(weights, data_filtered, subplot, title):
    x_min, x_max = np.min(data_filtered[:,0]), np.max(data_filtered[:,0])
    y_min, y_max = np.min(data_filtered[:,1]), np.max(data_filtered[:,1])
    z_min, z_max = np.min(data_filtered[:,2]), np.max(data_filtered[:,2])
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))
    zz = np.zeros_like(xx)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            point = [xx[i,j], yy[i,j], (z_min+z_max)/2]  # 初始化 z，可嘗試不同策略
            zz[i,j] = mlp_predict(weights, point)
    subplot.plot_surface(xx, yy, zz, alpha=0.3, cmap="Paired")
    subplot.scatter(data_filtered[:,0][data_filtered[:,-1]==1], data_filtered[:,1][data_filtered[:,-1]==1], data_filtered[:,2][data_filtered[:,-1]==1], c="red", label="Class 1")
    subplot.scatter(data_filtered[:,0][data_filtered[:,-1]==-1], data_filtered[:,1][data_filtered[:,-1]==-1], data_filtered[:,2][data_filtered[:,-1]==-1], c="blue", label="Class -1")
    subplot.set_title(title)
    subplot.set_xlabel("X")
    subplot.set_ylabel("Y")
    subplot.set_zlabel("Z")

def train(file_path, learning_rate, epochs, accuracy_limit, hidden_layers_size):
	data = np.loadtxt(file_path)
	dimension = np.size(data, 1) - 1
	layer_size = [dimension, *hidden_layers_size, 1]
	train_data, test_data = lib.preprocessing(data)

	# Train
	weights = []
	for i in range(len(layer_size) - 1):
		weights.append(np.random.randn(layer_size[i+1], layer_size[i]+1))

	for epoch in range(epochs):
		for row in train_data:
			# Feedforward
			y = [row[:-1]]
			for weight in weights:
				y.append(lib.sigmoid(weight @ [-1, *y[-1]]))

			# Backpropagation
			delta = [lib.delta_final(y[-1], lib.classify(row[-1]))]
			for layer in range(len(layer_size)-2, 0, -1):
				delta = [y[layer] * (1 - y[layer]) * (weights[layer].T[1:] @ delta[0]), *delta]
			for layer in range(len(weights)):
				weights[layer] += learning_rate * np.outer(delta[layer], [-1, *y[layer]])

		# Test
		correct = 0
		for row in test_data:
			y = row[:-1]
			for weight in weights:
				y = lib.sigmoid(weight @ [-1, *y])
			if(row[-1] == lib.classify(y)):
				correct += 1
		accuracy = correct / len(test_data)
		if(accuracy >= accuracy_limit):
			break

	plot = False
	if dimension == 2:
		plt.subplot(1, 2, 1)
		plot_decision_boundary_2d_mlp(weights, train_data, "Train")
		plt.subplot(1, 2, 2)
		plot_decision_boundary_2d_mlp(weights, test_data, "Test")
		plt.tight_layout()
		plot = True
	elif dimension == 3:
		subplot1 = plt.subplot(1, 2, 1, projection="3d")
		plot_decision_boundary_3d_mlp(weights, train_data, subplot1, "Train")
		subplot2 = plt.subplot(1, 2, 2, projection="3d")
		plot_decision_boundary_3d_mlp(weights, test_data, subplot2, "Test")
		plt.tight_layout()
		plot = True

	return epoch+1, accuracy, plot
