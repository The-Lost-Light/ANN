import numpy as np
import lib
import plot


def predict(weights):
	def predict_bind(x):
		y = x
		for w in weights:
			y = lib.sigmoid(w @ np.array([-1, *y]))
		return lib.classify(y)
	return predict_bind


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
			prediction =  predict(weights)(row[:-1])
			if(prediction == row[-1]):
				correct += 1
		accuracy = correct / len(test_data)
		if(accuracy >= accuracy_limit):
			break

	layout = plot.layout(data)
	plot.decision_boundary(predict(weights), (train_data, test_data), dimension, layout, mlp=True)

	return epoch+1, accuracy
