import numpy as np
from mlp import MLP
from model.geometry import Point2D, Line2D
import simulation


train_data = np.loadtxt("data/train4dAll.txt")
X = train_data[:, :-1]
y = train_data[:, -1].reshape(-1, 1)

mlp = MLP(hidden_layer_sizes=[8, 4, 2], learning_rate=0.01, epochs=100)
mlp.fit(X, y)

with open("data/軌道座標點.txt", "r") as f:
	first_line = f.readline().strip()
car_coordinate = [float(x) for x in first_line.split(",")]
area_data = np.loadtxt("data/軌道座標點.txt", delimiter=",", skiprows=1)
area_coordinate = area_data[2:]
finish_area_coordinate = [
	area_data[0],
	[area_data[1, 0], area_data[0, 1]],
	area_data[1],
	[area_data[0, 0], area_data[1, 1]],
]
start_line = Line2D(Point2D(-6, 0), Point2D(6, 0))
start_point = Point2D(0, 0)
simulation.run(area_coordinate, finish_area_coordinate, start_line, start_point)
