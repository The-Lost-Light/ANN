import numpy as np
import matplotlib.pyplot as plt
import os

def layout(data):
	dimension = data.shape[1] - 1
	mins = np.min(data[:, :dimension], axis=0)
	maxs = np.max(data[:, :dimension], axis=0)

	ranges = maxs - mins
	max_range = np.max(ranges) * 1.5

	centers = (mins + maxs) / 2
	outers = [(c - max_range/2, c + max_range/2) for c in centers]

	return tuple(centers), outers

def decision_boundary_2d(ax, predict, data, layout, title, slope=None, mlp=False):
	(x_center, y_center), [(x_min, x_max), (y_min, y_max)] = layout
	x, y, labels = data[:, 0], data[:, 1], data[:, -1]

	ax.scatter(x[labels == 1], y[labels == 1], s=10, c="red", label="Class 1")
	ax.scatter(x[labels == 0], y[labels == 0], s=10, c="blue", label="Class 0")

	if(slope != None):
		y_correspond = predict([x_center])
		axline = ax.axline((x_center, y_correspond), slope=slope, color="red", label="decision boundary")
	elif(mlp):
		xmg, ymg = np.meshgrid(
			np.arange(x_min, x_max, 0.1),
			np.arange(y_min, y_max, 0.1)
		)
		grid = np.c_[xmg.ravel(), ymg.ravel()]
		prediction = np.array([predict(p) for p in grid]).reshape(xmg.shape)
		ax.contourf(xmg, ymg, prediction, alpha=0.3, cmap="Paired")

	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_xlim(x_min, x_max)
	ax.set_ylim(y_min, y_max)
	ax.set_title(title)
	ax.legend()


def decision_boundary_3d(ax, predict, data, layout, title, mlp=False):
	_, [(x_min, x_max), (y_min, y_max), (z_min, z_max)] = layout
	x, y, z, labels = data[:, 0], data[:, 1], data[:, 2], data[:, -1]

	ax.scatter(x[labels==1], y[labels==1], z[labels==1], c="red", label="Class 1")
	ax.scatter(x[labels==0], y[labels==0], z[labels==0], c="blue", label="Class 0")

	if(not mlp):
		surface_x, surface_y = np.meshgrid(np.linspace(np.min(x), np.max(x), 10), np.linspace(np.min(y), np.max(y), 10))
		surface_z = predict([surface_x, surface_y])
		ax.plot_surface(surface_x, surface_y, surface_z, color="green", alpha=0.3)
	elif(mlp):
		xmg, ymg, zmg = np.meshgrid(
			np.arange(x_min, x_max, 0.1),
			np.arange(y_min, y_max, 0.1),
			np.arange(z_min, z_max, 0.1)
		)

		grid = np.c_[xmg.ravel(), ymg.ravel(), zmg.ravel()]
		prediction = np.array([predict(p) for p in grid])

		sc = ax.scatter(grid[:,0], grid[:,1], grid[:,2],c=prediction, cmap="coolwarm", alpha=0.1, s=5)
		fig = ax.get_figure()
		fig.colorbar(sc, ax=ax, shrink=0.5, aspect=10, label="Prediction")

	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	ax.set_title(title)
	ax.legend()


def decision_boundary(file_name, predict, data, dimension, layout, slope=None, mlp=False):
	train_data, test_data = data

	fig = None
	if dimension == 2 or dimension == 3:
		if dimension == 2:
			fig, (ax1, ax2) = plt.subplots(1, 2)
			decision_boundary_2d(ax1, predict, train_data, layout, "train", slope=slope, mlp=mlp)
			decision_boundary_2d(ax2, predict, test_data, layout, "test", slope=slope, mlp=mlp)
		elif dimension == 3:
			fig = plt.figure()
			ax1 = plt.subplot(1, 2, 1, projection="3d")
			decision_boundary_3d(ax1, predict, train_data, layout, "train", mlp=mlp)
			ax2 = plt.subplot(1, 2, 2, projection="3d")
			decision_boundary_3d(ax2, predict, test_data, layout, "test", mlp=mlp)

		if(fig):
			fig.tight_layout()
			os.makedirs("plots", exist_ok=True)
			fig.savefig("plots/" + file_name + ("-mlp.png" if mlp else ".png"))

	return fig
