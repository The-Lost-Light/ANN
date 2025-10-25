import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from model.object import Car, Ground


class Simulation:
	def __init__(self, area, finish_area, start_line, start_point):
		self.ground = Ground(area, finish_area, start_line)
		self.car = Car(start_point)

	def run(self, mlp):
		area = self.ground.area
		finish_area = self.ground.finish_area

		while True:
			camera_position = self.car.camera_position
			if all(not area.inArea(p) for p in camera_position):
				break

			camera_direction = self.car.camera_direction
			distance = [self.ground.area.ray_intersection(camera_position[i], camera_direction[i]) for i in range(3)]

			wheelAngle = mlp.predict(distance)
			self.car.move(wheelAngle)

			position = self.car.position[-1]
			if finish_area.inArea(position) or not area.inArea(position):
				break
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
		for i in range(len(self.car.position)):
			ax.plot([p.x for p in self.car.position][: i + 1], [p.y for p in self.car.position][: i + 1], c="blue")
			plt.pause(0.1)

		plt.show()
