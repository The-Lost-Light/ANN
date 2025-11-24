import numpy as np


class HopfieldNetwork:
	def __init__(self, pattern_size):
		self.pattern_size = pattern_size
		self.weights = np.zeros((pattern_size, pattern_size))

	def train(self, patterns, threadhold=False):
		N = len(patterns)
		if N == 0:
			return

		for p in patterns:
			self.weights += np.outer(p, p.T)

		self.weights -= N * np.identity(self.pattern_size)
		self.weights /= self.pattern_size
		self.theta = np.sum(self.weights, axis=0).reshape(-1, 1) if threadhold else np.zeros((len(self.weights), 1))

	def _update_unit(self, x, i=None):
		if i is not None:
			v = self.weights[i] @ x - self.theta[i]
		else:
			v = self.weights @ x - self.theta
		for j in range(len(v)):
			if v[j] > 0:
				v[j] = 1
			elif v[j] == 0:
				v[j] = x[j]
			elif v[j] < 0:
				v[j] = -1
		return v

	def predict(self, pattern, max_iter=1000, asynchronous=False):
		x = pattern.copy()
		for iter in range(max_iter):
			old_x = x.copy()
			if asynchronous:
				i = np.random.randint(0, self.pattern_size)
				x[i] = self._update_unit(x, i)[0]
			else:
				x = self._update_unit(x)
			if np.array_equal(x, old_x) and iter >= 100:
				break
		return x
