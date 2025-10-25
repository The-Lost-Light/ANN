import numpy as np
from mlp import MLP
from simulation import Simulation

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
(x1, y1), (x2, y2) = area_data[0], area_data[1]
finish_area_coordinate = [
	[x1, y1],
	[x2, y1],
	[x2, y2],
	[x1, y2],
]
start_line = [[-6, 0], [6, 0]]
start_point = [0, 0]

simulation = Simulation(area_coordinate, finish_area_coordinate, start_line, start_point)
simulation.run(mlp)
