import numpy as np
import matplotlib.pyplot as plt

def layout(data):
	dimension = data.shape[1] - 1
	mins = np.min(data[:, :dimension], axis=0)
	maxs = np.max(data[:, :dimension], axis=0)

	ranges = maxs - mins
	max_range = np.max(ranges) * 1.5

	centers = (mins + maxs) / 2
	outers = [(c - max_range/2, c + max_range/2) for c in centers]

	return tuple(centers), outers

def decision_boundary_2d(predict, data, layout, title, slope=None, mlp=False):
	(x_center, y_center), [(x_min, x_max), (y_min, y_max)] = layout
	x, y, labels = data[:, 0], data[:, 1], data[:, -1]

	plt.scatter(x[labels == 1], y[labels == 1], s=10, c="red", label="Class 1")
	plt.scatter(x[labels == 0], y[labels == 0], s=10, c="blue", label="Class 0")

	if(slope != None):
		y_correspond = predict([x_center])
		axline = plt.axline((x_center, y_correspond), slope=slope, color="red", label="decision boundary")
	elif(mlp):
		xmg, ymg = np.meshgrid(
			np.arange(x_min, x_max, 0.1),
			np.arange(y_min, y_max, 0.1)
		)
		grid = np.c_[xmg.ravel(), ymg.ravel()]
		prediction = np.array([predict(p) for p in grid]).reshape(xmg.shape)
		plt.contourf(xmg, ymg, prediction, alpha=0.3, cmap="Paired")

	plt.xlabel('X')
	plt.ylabel('Y')
	plt.xlim(x_min, x_max)
	plt.ylim(y_min, y_max)
	plt.title(title)
	plt.legend()


def decision_boundary_3d(predict, data, layout, subplot, title, mlp=False):
	_, [(x_min, x_max), (y_min, y_max), (z_min, z_max)] = layout
	x, y, z, labels = data[:, 0], data[:, 1], data[:, 2], data[:, -1]

	subplot.scatter(x[labels==1], y[labels==1], z[labels==1], c="red", label="Class 1")
	subplot.scatter(x[labels==0], y[labels==0], z[labels==0], c="blue", label="Class 0")

	if(not mlp):
		surface_x, surface_y = np.meshgrid(np.linspace(np.min(x), np.max(x), 10), np.linspace(np.min(y), np.max(y), 10))
		surface_z = predict([surface_x, surface_y])
		subplot.plot_surface(surface_x, surface_y, surface_z, color="green", alpha=0.3)
	elif(mlp):
		xmg, ymg, zmg = np.meshgrid(
			np.arange(x_min, x_max, 0.1),
			np.arange(y_min, y_max, 0.1),
			np.arange(z_min, z_max, 0.1)
		)

		grid = np.c_[xmg.ravel(), ymg.ravel(), zmg.ravel()]
		prediction = np.array([predict(p) for p in grid])

		sc = subplot.scatter(grid[:,0], grid[:,1], grid[:,2],c=prediction, cmap="coolwarm", alpha=0.1, s=5)
		plt.colorbar(sc, ax=subplot, shrink=0.5, aspect=10, label="Prediction")

	subplot.set_xlabel('X')
	subplot.set_ylabel('Y')
	subplot.set_zlabel('Z')
	subplot.set_title(title)
	subplot.legend()


def decision_boundary(predict, data, dimension, layout, slope=None, mlp=False):
	train_data, test_data = data

	if dimension == 2 or dimension == 3:
		if dimension == 2:
			plt.subplot(1, 2, 1)
			decision_boundary_2d(predict, train_data, layout, "train", slope=slope, mlp=mlp)
			plt.subplot(1, 2, 2)
			decision_boundary_2d(predict, test_data, layout, "test", slope=slope, mlp=mlp)
		elif dimension == 3:
			subplot1 = plt.subplot(1, 2, 1, projection="3d")
			decision_boundary_3d(predict, train_data, layout, subplot1, "train", mlp=mlp)
			subplot2 = plt.subplot(1, 2, 2, projection="3d")
			decision_boundary_3d(predict, test_data, layout, subplot2, "test", mlp=mlp)

		plt.tight_layout()
		plt.savefig("cache/plot.png")
		plt.show()
		plt.close()
