import math
from model.geometry import Point2D, Line2D, Polygon2D


def angle_to_vector(angle):
	radian = math.radians(angle)
	return Point2D(math.cos(radian), math.sin(radian))


class Car:
	def __init__(
		self,
		initial_position=[0, 0],
		initial_angle=90,
		diameter=6,
		angle_min=-90,
		angle_max=270,
		wheel_min=-40,
		wheel_max=40,
	):
		self.initial_position = Point2D.from_list(initial_position)
		self.initial_angle = initial_angle
		self.radius = diameter / 2
		self.angle_min = angle_min
		self.angle_max = angle_max
		self.wheel_min = wheel_min
		self.wheel_max = wheel_max

		self.reset()

	def reset(self):
		self.position = [self.initial_position]
		self.angle = self.initial_angle
		self.wheelAngle = 0

	@property
	def wheelAngle(self):
		return self._wheelAngle

	@wheelAngle.setter
	def wheelAngle(self, value):
		if value < self.wheel_min:
			value = self.wheel_min
		elif value > self.wheel_max:
			value = self.wheel_max
		self._wheelAngle = value

	@property
	def angle(self):
		return self._angle

	@angle.setter
	def angle(self, value):
		if value > self.angle_max:
			value -= self.angle_max - self.angle_min
		self._angle = value

	@property
	def camera_position(self) -> list[Point2D]:
		return [self.position[-1] + self.radius * angle_to_vector(self.angle + delta) for delta in (0, -45, 45)]

	@property
	def camera_direction(self) -> list[Point2D]:
		return [angle_to_vector(self.angle + delta) for delta in (0, -45, 45)]

	def move(self, wheelAngle=None):
		if wheelAngle is not None:
			self.wheelAngle = wheelAngle

		angle_r = math.radians(self.angle)
		wheelAngle_r = math.radians(self.wheelAngle)
		x = self.position[-1].x + math.cos(angle_r + wheelAngle_r) + math.sin(wheelAngle_r) * math.sin(angle_r)
		y = self.position[-1].y + math.sin(angle_r + wheelAngle_r) - math.sin(wheelAngle_r) * math.cos(angle_r)
		angle = (self.angle - math.degrees(math.asin(math.sin(wheelAngle_r) / self.radius))) % 360

		self.position.append(Point2D(x, y))
		self.angle = angle


class Ground:
	def __init__(self, area_points, finish_area_points, start_line_points):
		self.area = Polygon2D.from_list(area_points)
		self.finish_area = Polygon2D.from_list(finish_area_points)
		self.start_line = Line2D.from_list(start_line_points)
