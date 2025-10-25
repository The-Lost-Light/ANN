from __future__ import annotations
import math
import numpy as np


class Point2D:
	def __init__(self, x: float, y: float):
		self.x = x
		self.y = y

	@classmethod
	def from_list(cls, list: list[float]) -> Point2D:
		return cls(list[0], list[1])

	def __str__(self) -> str:
		return f"({self.x}, {self.y})"

	def __repr__(self) -> str:
		return f"Point2D{self.__str__()}"

	def __add__(self, point: Point2D) -> Point2D:
		x = self.x + point.x
		y = self.y + point.y
		return Point2D(x, y)

	def __sub__(self, point: Point2D) -> Point2D:
		x = self.x - point.x
		y = self.y - point.y
		return Point2D(x, y)

	def __mul__(self, scale: float) -> Point2D:
		return Point2D(scale * self.x, scale * self.y)

	def __rmul__(self, scale: float) -> Point2D:
		return self.__mul__(scale)

	def __iter__(self):
		yield self.x
		yield self.y

	def __array__(self, dtype=None) -> np.ndarray:
		return np.asarray(list(self))

	def length(self) -> float:
		return self.distance(Point2D(0, 0))

	def dot(self, point: Point2D) -> float:
		return self.x * point.x + self.y * point.y

	def cross(self, point: Point2D) -> float:
		return self.x * point.y - self.y * point.x

	def distance(self, object: Point2D | Line2D) -> float:
		if isinstance(object, Point2D):
			difference = self - object
			return math.hypot(difference.x, difference.y)
		elif isinstance(object, Line2D):
			vector_point = self - object.point1
			vector_line = object.point2 - object.point1
			projection_ratio = vector_point.dot(vector_line) / object.length() ** 2

			if 0 <= projection_ratio <= 1:
				projection_point = object.point1 + projection_ratio * (object.point2 - object.point1)
				return self.distance(projection_point)
			else:
				return min(self.distance(object.point1), self.distance(object.point2))


class Line2D:
	def __init__(self, point1: Point2D, point2: Point2D):
		self.point1 = point1
		self.point2 = point2

	@classmethod
	def from_list(cls, list: list[list[float]]) -> Line2D:
		return cls(Point2D.from_list(list[0]), Point2D.from_list(list[1]))

	def __str__(self) -> str:
		return f"[({self.point1.x}, {self.point1.y}), ({self.point2.x}, {self.point2.y})]"

	def __repr__(self) -> str:
		return f"Line2D{self.__str__()}"

	def __iter__(self):
		yield list(self.point1)
		yield list(self.point2)

	def __array__(self, dtype=None) -> np.ndarray:
		return np.asarray(list(self))

	def length(self) -> float:
		return self.point1.distance(self.point2)

	def isLeft(self, point: Point2D) -> float:
		return point.cross(self.point1) + self.point1.cross(self.point2) + self.point2.cross(point)


class Polygon2D:
	def __init__(self, edges: list[Line2D]):
		self.edges = edges

	@classmethod
	def from_Point2Ds(cls, points: list[Point2D]) -> Polygon2D:
		edges = []
		for i in range(len(points) - 1):
			edges.append(Line2D(points[i], points[i + 1]))
		return cls(edges)

	@classmethod
	def from_list(cls, points: list[list[float]]) -> Polygon2D:
		point2Ds = [Point2D.from_list(p) for p in points]
		return cls.from_Point2Ds(point2Ds)

	def __iter__(self):
		for edges in self.edges:
			yield list(edges.point1)
		yield list(edges.point2)

	def __array__(self, dtype=None):
		return np.asarray(list(self), dtype=dtype)

	def inArea(self, point: Point2D) -> bool:
		winding_number = 0
		for edge in self.edges:
			if edge.point1.y <= point.y < edge.point2.y and edge.isLeft(point) > 0:
				winding_number += 1
			elif edge.point2.y < point.y <= edge.point1.y and edge.isLeft(point) < 0:
				winding_number -= 1
		return winding_number != 0

	def ray_intersection(self, point: Point2D, direction: Point2D) -> float:
		min_scaler = float("inf")

		for edge in self.edges:
			edge_direction = edge.point2 - edge.point1
			denominator = direction.cross(edge_direction)

			if abs(denominator) < 1e-9:
				continue

			v = edge.point1 - point
			direction_scaler = v.cross(edge_direction) / denominator
			edge_scaler = v.cross(direction) / denominator

			if 0 <= direction_scaler < min_scaler and 0 <= edge_scaler <= 1:
				min_scaler = direction_scaler

		return min_scaler * direction.length()
