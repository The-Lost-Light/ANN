import numpy as np
import matplotlib.pyplot as plt
import lib

def plot_decision_boundary_2d(w, data_filtered, title, x_center, y_center, max_range):
	x = data_filtered[:, 0]
	y = data_filtered[:, 1]
	labels = data_filtered[:, -1]

	plt.scatter(x[labels == 1], y[labels == 1], s=10, c="red", label="Class 1")
	plt.scatter(x[labels == 0], y[labels == 0], s=10, c="blue", label="Class 0")

	if w[2] != 0:
		slope = -w[1] / w[2]
		y_correspond = (w[0] - w[1] * x_center) / w[2]
		axline = plt.axline((x_center, y_correspond), slope=slope, color="red", label="decision boundary")
	else:
		axline = plt.axline((x_center, 0), slope=np.inf, color="red", label="decision boundary")


	plt.xlabel('X')
	plt.ylabel('Y')
	plt.xlim(x_center - max_range/2, x_center + max_range/2)
	plt.ylim(y_center - max_range/2, y_center + max_range/2)
	plt.title(title)
	plt.legend()


def plot_decision_boundary_3d(w, data_filtered, subplot, title):
	x = data_filtered[:, 0]
	y = data_filtered[:, 1]
	z = data_filtered[:, 2]
	labels = data_filtered[:, 3]

	subplot.scatter(x[labels==1], y[labels==1], z[labels==1], c="red", label="Class 1")
	subplot.scatter(x[labels==0], y[labels==0], z[labels==0], c="blue", label="Class 0")

	surface_x, surface_y = np.meshgrid(np.linspace(np.min(x), np.max(x), 10), np.linspace(np.min(y), np.max(y), 10))
	if w[3] != 0:
		surface_z = (-w[0] - w[1] * surface_x - w[2] * surface_y) / w[3]
		subplot.plot_surface(surface_x, surface_y, surface_z, color="green", alpha=0.3)

	subplot.set_xlabel('X')
	subplot.set_ylabel('Y')
	subplot.set_zlabel('Z')
	subplot.set_title(title)
	subplot.legend()


def train(file_path, learning_rate, epochs, accuracy_limit):
	data = np.loadtxt(file_path)
	dimension = np.size(data, 1) - 1
	train_data, test_data = lib.preprocessing(data)

	# Train
	w = np.array([-1, *np.zeros(dimension - 1), 1])

	for epoch in range(epochs):
		for row in train_data:
			input = np.array([-1, *row[:-1]])
			if w @ input < 0 and row[-1] == 1:
				w = w + learning_rate * input
			elif w @ input > 0 and row[-1] == 0:
				w = w - learning_rate * input

		# Test
		correct = 0
		for row in test_data:
			input = np.array([-1, *row[:-1]])
			if (w @ input >= 0 and row[-1] == 1) or (w @ input < 0 and row[-1] == 0):
				correct += 1
		accuracy = correct / len(test_data)
		if(accuracy >= accuracy_limit):
			break


	# plot
	x_min, x_max = np.min(data[:, 0]), np.max(data[:, 0])
	y_min, y_max = np.min(data[:, 1]), np.max(data[:, 1])
	max_range = max(x_max - x_min, y_max - y_min) * 1.5
	x_center = (x_min + x_max) / 2
	y_center = (y_min + y_max) / 2

	plot = False
	if dimension == 2:
		plt.subplot(1, 2, 1)
		plot_decision_boundary_2d(w, train_data, "train", x_center, y_center, max_range)
		plt.subplot(1, 2, 2)
		plot_decision_boundary_2d(w, test_data, "test", x_center, y_center, max_range)
		plt.tight_layout()
		plot = True
	elif dimension == 3:
		subplot1 = plt.subplot(1, 2, 1, projection="3d")
		plot_decision_boundary_3d(w, train_data, subplot1, "train")
		subplot2 = plt.subplot(1, 2, 2, projection="3d")
		plot_decision_boundary_3d(w, test_data, subplot2, "test")
		plt.tight_layout()
		plot = True

	return epoch+1, accuracy, plot
