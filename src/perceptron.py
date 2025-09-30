import numpy as np
import lib
import plot


def get_slope(weight):
	return (-weight[1] / weight[2]) if(weight[-1] != 0) else np.inf


def predict(weight):
	def predict_bind(x):
		if(weight[-1] != 0):
			y = weight[0]
			for i in range(len(x)):
				y -= weight[i+1] * x[i]
			y /= weight[-1]
		else:
			y = 0
		return y
	return predict_bind


def train(file_path, file_name, learning_rate, epochs, accuracy_limit):
	data = np.loadtxt(file_path)
	dimension = np.size(data, 1) - 1
	train_data, test_data = lib.preprocessing(data)

	# Train
	weight = np.array([-1, *np.zeros(dimension - 1), 1])

	for epoch in range(epochs):
		for row in train_data:
			input = np.array([-1, *row[:-1]])
			if weight @ input < 0 and row[-1] == 1:
				weight = weight + learning_rate * input
			elif weight @ input > 0 and row[-1] == 0:
				weight = weight - learning_rate * input

		# Test
		correct = 0
		for row in test_data:
			input = np.array([-1, *row[:-1]])
			if (weight @ input >= 0 and row[-1] == 1) or (weight @ input < 0 and row[-1] == 0):
				correct += 1
		accuracy = correct / len(test_data)
		if(accuracy >= accuracy_limit):
			break

	# plot
	layout = plot.layout(data)
	slope= get_slope(weight) if dimension == 2 else None
	fig = plot.decision_boundary(file_name, predict(weight), (train_data, test_data), dimension, layout, slope=slope)

	return epoch+1, accuracy, weight, fig
