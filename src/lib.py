import numpy as np
import plot

def preprocessing(file_path):
	data = np.loadtxt(file_path)

	labels = data[:, -1]
	labels -= np.min(labels)

	dimension = np.size(data, 1) - 1
	label_unique = np.unique(data[:, -1])
	layout = plot.layout(data)

	if (len(data) >= 10):
		indices = np.arange(len(data))
		np.random.shuffle(indices)
		train_size = int(len(data) * 2 / 3)
		train_indices, test_indices = indices[:train_size], indices[train_size:]
		train_data, test_data = data[train_indices], data[test_indices]
	else:
		train_data, test_data = data, data
	return train_data, test_data , dimension, label_unique, layout


def classify(label):
	if(label >= 0.5):
		return 1
	else:
		return 0
