import numpy as np
import lib
import plot


def sigmoid(x):
	return 1 / (1 + np.exp(-x))


def sigmoid_derive(x):
	return sigmoid(x) * (1 - sigmoid(x))


def parse_labels(labels, label_uniques):
	return np.eye(1, label_uniques, int(labels))[0] if label_uniques > 2 else labels


def delta_final(prediction, output):
	return (output - prediction) * prediction * (1 - prediction)


def predict(weights, label_uniques):
	def predict_bind(x):
		y = x
		for w in weights:
			y = sigmoid(w @ np.array([-1, *y]))
		if(label_uniques > 2):
			return y.argmax()
		else:
			return lib.classify(y)
	return predict_bind


def train(file_path, file_name, learning_rate, epochs, accuracy_limit, hidden_layers_size):
	train_data, test_data , dimension, label_unique, layout = lib.preprocessing(file_path)
	label_uniques = len(label_unique)
	output_size = label_uniques if label_uniques > 2 else 1
	layer_size = [dimension, *hidden_layers_size, output_size]

	# Train
	weights = []
	for i in range(len(layer_size) - 1):
		weights.append(np.random.randn(layer_size[i+1], layer_size[i]+1))

	for epoch in range(epochs):
		for row in train_data:
			# Feedforward
			y = [row[:-1]]
			for weight in weights:
				y.append(sigmoid(weight @ [-1, *y[-1]]))

			# Backpropagation
			delta = [delta_final(y[-1], parse_labels(row[-1], label_uniques))]
			for layer in range(len(layer_size)-2, 0, -1):
				delta = [y[layer] * (1 - y[layer]) * (weights[layer].T[1:] @ delta[0]), *delta]
			for layer in range(len(weights)):
				weights[layer] += learning_rate * np.outer(delta[layer], [-1, *y[layer]])

		# Test
		correct = 0
		for row in test_data:
			prediction = predict(weights, label_uniques)(row[:-1])
			if(prediction == row[-1]):
				correct += 1
		accuracy = correct / len(test_data)
		if(accuracy >= accuracy_limit):
			break

	fig = plot.decision_boundary(file_name, predict(weights, label_uniques), (train_data, test_data), dimension, layout, label_unique, mlp=True)

	return epoch+1, accuracy, weights, fig
