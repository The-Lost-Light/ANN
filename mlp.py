import numpy as np


class MLP:
	def __init__(self, hidden_layer_sizes=(20,12), learning_rate=0.001, epochs=100):
		self.hidden_layer_sizes = hidden_layer_sizes
		self.learning_rate = learning_rate
		self.epochs = epochs

	def _initialize_weights(self, n_features, n_outputs):
		layer_sizes = [n_features] + list(self.hidden_layer_sizes) + [n_outputs]
		self.weights = []

		for i in range(len(layer_sizes) - 1):
			W = np.random.randn(layer_sizes[i + 1], layer_sizes[i] + 1) * np.sqrt(2 / layer_sizes[i])
			self.weights.append(W)

	def _activation(self, v):
		return np.maximum(0, v)

	def _activation_derive(self, v):
		return (v > 0).astype(int)

	def _forward(self, input):
		activations = [input]
		for i in range(len(self.weights) - 1):
			x = np.insert(activations[-1], 0, -1).reshape(-1, 1)
			v = self.weights[i] @ x
			y = self._activation(v)
			activations.append(y)
		x = np.insert(activations[-1], 0, -1).reshape(-1, 1)
		v = self.weights[-1] @ x
		activations.append(v)
		return activations

	def _backward(self, activations, y):
		grads_W = [None] * len(self.weights)

		delta = (y - activations[-1]) * self._activation_derive(activations[-1])
		activation_with_bias = np.insert(activations[-2], 0, -1).reshape(-1, 1)
		grads_W[-1] = np.outer(delta, activation_with_bias)

		for i in reversed(range(len(self.weights) - 1)):
			delta = self._activation_derive(activations[i + 1]) * (self.weights[i + 1].T[1:] @ delta)
			activation_with_bias = np.insert(activations[i], 0, -1).reshape(-1, 1)
			grads_W[i] = np.outer(delta, activation_with_bias)

		return grads_W

	def fit(self, X, y):
		X = np.asarray(X)
		y = np.asarray(y).reshape(-1, 1)

		n_samples, n_features = X.shape
		n_outputs = y.shape[1]
		self._initialize_weights(n_features, n_outputs)

		for epoch in range(self.epochs):
			for sample_index in range(len(X)):
				activations = self._forward(X[sample_index])
				grads_W = self._backward(activations, y[sample_index])

				for i in range(len(self.weights)):
					self.weights[i] += self.learning_rate * grads_W[i]

	def predict(self, x):
		x = np.asarray(x)
		activations = self._forward(x)
		return activations[-1][0, 0]
