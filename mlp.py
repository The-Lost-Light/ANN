import numpy as np


class MLP:
	def __init__(self, hidden_layer_sizes=[8,4], learning_rate=0.001, epochs=100):
		self.hidden_layer_sizes = hidden_layer_sizes
		self.learning_rate = learning_rate
		self.epochs = epochs

	def _standardize(self, X):
		self.mean = np.mean(X, axis=0)
		self.std = np.std(X, axis=0)

	def _standardize_transform(self, X):
		return (X - self.mean) / self.std

	def _minmax(self, min, max):
		self.y_min = min
		self.y_max = max

	def _minmax_transform(self, y):
		return (y - self.y_min) / (self.y_max - self.y_min)

	def _minmax_transform_inverse(self, y):
		return y * (self.y_max - self.y_min) + self.y_min

	def _initialize_weights(self, n_features, n_outputs):
		layer_sizes = [n_features, *self.hidden_layer_sizes, n_outputs]
		self.weights = []

		for i in range(len(layer_sizes) - 1):
			self.weights.append(np.random.randn(layer_sizes[i + 1], layer_sizes[i] + 1))

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

		delta = (y - activations[-1])
		grads_W[-1] = np.outer(delta.reshape(-1), np.insert(activations[-2], 0, -1))

		for i in reversed(range(len(self.weights) - 1)):
			delta = self._activation_derive(activations[i + 1]) * (self.weights[i + 1][:, 1:].T @ delta)
			grads_W[i] = np.outer(delta.reshape(-1), np.insert(activations[i], 0, -1))

		return grads_W

	def fit(self, X, y):
		X = np.asarray(X)
		y = np.asarray(y)

		self._standardize(X)
		X = self._standardize_transform(X)
		self._minmax(-40, 40)
		y = self._minmax_transform(y)

		n_samples, n_features = X.shape
		n_outputs = y.shape[1]
		self._initialize_weights(n_features, n_outputs)

		for epoch in range(self.epochs):
			for sample_index in range(n_samples):
				activations = self._forward(X[sample_index])
				grads_W = self._backward(activations, y[sample_index])

				for i in range(len(self.weights)):
					self.weights[i] += self.learning_rate * grads_W[i]

	def get_weights(self):
		return self.weights

	def predict(self, x):
		x = np.asarray(x)
		self._standardize_transform(x)
		prediction = self._forward(x)[-1][0, 0]
		return self._minmax_transform_inverse(prediction)
