import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def run(area, finish_area, start_line, start_point):
	fig, ax = plt.subplots()

	polygon = Polygon(area, edgecolor="lightblue", linewidth=2, fill=False)
	finish_polygon = Polygon(finish_area, edgecolor="red", linewidth=1, facecolor="lightcoral")

	ax.add_patch(polygon)
	ax.add_patch(finish_polygon)
	start_line = np.asarray(start_line).T
	ax.plot(start_line[0], start_line[1], zorder=0)
	ax.scatter(start_point.x, start_point.y, c="black")

	ax.set_aspect("equal")
	ax.set_title("Simulation")
	plt.show()
