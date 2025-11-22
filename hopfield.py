import numpy as np


class HopfieldNetwork:
	def __init__(self, pattern_size):
		self.pattern_size = pattern_size
		self.weights = np.zeros((pattern_size, pattern_size))

	def _network_recall(self, x):
		v = self.weights @ x - self.theta
		for j in range(len(v)):
			if v[j] > 0:
				v[j] = 1
			elif v[j] == 0:
				v[j] = x[j]
			elif v[j] < 0:
				v[j] = -1
		return v

	def train(self, patterns):
		N = len(patterns)
		if N == 0:
			return

		for p in patterns:
			self.weights += np.dot(p, p.T)

		self.weights -= N * np.identity(self.pattern_size)
		self.weights /= self.pattern_size
		self.theta = np.sum(self.weights, axis=0).reshape(-1, 1)

	def predict(self, pattern, max_iter=100):
		recall_old = None
		recall = pattern
		iter = 0
		while (recall != recall_old).any() and iter < max_iter:
			recall_old = recall
			recall = self._network_recall(recall_old)
			iter += 1
		return recall
