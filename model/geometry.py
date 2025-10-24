import math
import numpy as np


class Point2D:
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def __add__(self, point):
		x = self.x + point.x
		y = self.y + point.y
		return Point2D(x, y)

	def __sub__(self, point):
		x = self.x - point.x
		y = self.y + point.y
		return Point2D(x, y)

	def __mul__(self, scale):
		return Point2D(scale * self.x, scale * self.y)

	def __rmul__(self, scale):
		return self.__mul__(scale)

	def __iter__(self):
		yield self.x
		yield self.y

	def __array__(self, dtype=None):
		return np.asarray(list(self))

	def distance(self, object):
		if isinstance(object, Point2D):
			difference = self - object
			return math.hypot(difference.x, difference.y)
		elif isinstance(object, Line2D):
			vector_point = self - object.point1
			vector_line = object.point2 - object.point1
			projection_ratio = (vector_point.x * vector_line.x + vector_point.y * vector_line.y) / len(object) ** 2

			if 0 <= projection_ratio <= 1:
				projection_point = object.point1 + projection_ratio * (object.point2 - object.point1)
				return self.distance(projection_point)
			else:
				return min(self.distance(object.point1), self.distance(object.point2))


class Line2D:
	def __init__(self, point1, point2):
		self.point1 = point1
		self.point2 = point2

	def __iter__(self):
		yield list(self.point1)
		yield list(self.point2)

	def __array__(self, dtype=None):
		return np.asarray(list(self))

	def __len__(self):
		return self.point1.distance(self.point2)

	def isLeft(self, point):
		return (self.point2.x - self.point1.x) * (point.y - self.point1.y) - (point.x - self.point1.x) * (
			self.point2.y - self.point1.y
		)


class Polygon2D:
	def __init__(self, points):
		self.edges = []
		for i in range(len(points) - 1):
			self.edges.append(Line2D(points[i], points[i + 1]))

	def inArea(self, point):
		winding_number = 0
		for e in self.edges:
			if e.point1.y <= point.y < e.point2.y and e.is_left(point) > 0:
				winding_number += 1
			elif e.point2.y < point.y <= e.point1.y and e.is_left(point) < 0:
				winding_number -= 1
		return winding_number != 0
