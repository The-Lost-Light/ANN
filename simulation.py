import numpy as np
from mlp import MLP
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from model.object import Car, Ground


class Simulation:
	def __init__(self, area, finish_area, start_line, initial_position, initial_angle):
		self.ground = Ground(area, finish_area, start_line)
		self.car = Car(initial_position, initial_angle)

	@classmethod
	def from_track(cls, data_path):
		with open(data_path, "r") as f:
			first_line = f.readline().strip()

		car_x, car_y, car_angle = [float(x) for x in first_line.split(",")]

		area_data = np.loadtxt("data/軌道座標點.txt", delimiter=",", skiprows=1)
		area = area_data[2:]
		(x1, y1), (x2, y2) = area_data[0], area_data[1]
		finish_area = [
			[x1, y1],
			[x2, y1],
			[x2, y2],
			[x1, y2],
		]
		start_line = [[-6, 0], [6, 0]]

		return cls(area, finish_area, start_line, [car_x, car_y], car_angle)

	def _load_train_data(self, data_path):
		train_data = np.loadtxt(data_path)
		X = train_data[:, :-1]
		y = train_data[:, -1].reshape(-1, 1)
		return X, y

	def train(self, data_path, **train_parameters):
		self.mlp = MLP(**train_parameters)
		X, y = self._load_train_data(data_path)
		self.dimension = X.shape[1]
		self.mlp.fit(X, y)

	def run(self):
		area = self.ground.area
		finish_area = self.ground.finish_area
		self.distance = []

		while True:
			position = self.car.position[-1]
			camera_position = self.car.camera_position
			camera_direction = self.car.camera_direction
			self.distance.append([area.ray_intersection(camera_position[i], camera_direction[i]) for i in range(3)])

			if finish_area.inArea(position) or all(not area.inArea(p) for p in [position, *camera_position]):
				break

			input = self.distance[-1] if self.dimension == 3 else [*self.car.position[-1], *self.distance[-1]]
			wheelAngle = self.mlp.predict(input)
			self.car.move(wheelAngle)

		self.plot()

	def plot(self):
		fig, ax = plt.subplots()

		ax.set_aspect("equal")
		ax.set_title("Simulation")

		polygon = Polygon(list(self.ground.area), edgecolor="lightblue", linewidth=2, fill=False)
		finish_polygon = Polygon(list(self.ground.finish_area), edgecolor="red", linewidth=1, facecolor="lightcoral")

		ax.add_patch(polygon)
		ax.add_patch(finish_polygon)
		start_line = np.asarray(self.ground.start_line).T
		ax.plot(start_line[0], start_line[1], zorder=0)
		text_handle = ax.text(0.02, 0.99, "", transform=ax.transAxes, va="top", ha="left", fontsize=12, color="black")

		for i in range(len(self.car.position)):
			ax.plot([p.x for p in self.car.position][: i + 1], [p.y for p in self.car.position][: i + 1], c="blue")
			distances = self.distance[i]
			text_handle.set_text(f"Front: {distances[0]:.2f}\nRight: {distances[1]:.2f}\nLeft: {distances[2]:.2f}")
			plt.pause(0.1)

		plt.show()
