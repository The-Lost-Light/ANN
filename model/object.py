from model.geometry import Polygon2D


class Car:
	def __init__(self, radius=6, angle_min=-90, angle_max=270, wheel_min=-40, wheel_max=40):
		self.radius = radius
		self.angle_min = angle_min
		self.angle_max = angle_max
		self.wheel_min = wheel_min
		self.wheel_max = wheel_max

		self.reset()

	def reset(self, angle=90, speed=1):
		self.speed = speed
		self.angle = angle

	@property
	def wheelAngle(self):
		return self.wheelAngle

	@wheelAngle.setter
	def wheelAngle(self, value):
		if value < self.wheel_min:
			value = self.wheel_min
		elif value > self.wheel_max:
			value = self.wheel_max
		self.wheelAngle = value

	@property
	def angle(self):
		return self.angle

	@angle.setter
	def angle(self, value):
		if value < self.angle_min:
			value = self.angle_min
		elif value > self.angle_max:
			value = self.angle_max
		self.angle = value


class Ground:
	def __init__(self, area_points, finish_area_points):
		self.area = Polygon2D(area_points)
		self.finish_area = Polygon2D(finish_area_points)
