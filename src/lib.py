import numpy as np

def preprocessing(data):
	labels = data[:, -1]
	labels -= np.min(labels)

	if (len(data) >= 10):
		indices = np.arange(len(data))
		np.random.shuffle(indices)
		train_size = int(len(data) * 2 / 3)
		train_indices, test_indices = indices[:train_size], indices[train_size:]
		return data[train_indices], data[test_indices]
	else:
		return data, data


def classify(label):
	if(label >= 0.5):
		return 1
	else:
		return 0
