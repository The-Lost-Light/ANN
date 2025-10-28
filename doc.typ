#import "@preview/cheq:0.3.0": checklist
#show: checklist.with(fill: luma(95%), stroke: blue, radius: .2em)
#set heading(numbering: "一.1.a")

#let assets = "doc_assets"

= 加分題完成功能
- [x] 自行編寫模擬程式
= 程式介面說明
#image("assets/gui.png")
兩個選項可以選擇用`train4dAll.txt`或是`train6dAll.txt`訓練
= 實驗結果
#grid(
	gutter: 2em,
	block(align(center)[
		#image("assets/track4D.png")
		#text(0.8em, "train4dAll.txt")
	]),
	block(align(center)[
		#image("assets/track6D.png")
		#text(0.8em, "train6dAll.txt")
	])
)
= 程式說明
== 多層感知機
=== Min-Max
```python
def _minmax(self, X, min, max):
	self.min = (np.min(X, axis=0), min)
	self.max = (np.max(X, axis=0), max)

def _minmax_transform(self, x, index):
	return (x - self.min[index]) / (self.max[index] - self.min[index])

def _minmax_transform_inverse(self, x):
	return x * (self.max[1] - self.min[1]) + self.min[1]
```
$x'=(x-min)/(max-min)$

$x = x' times (max - min) + min$
=== 前饋(Forward)
```python
def _forward(self, input):
	activations = [input]
	for i in range(len(self.weights) - 1):
		x = np.insert(activations[-1], 0, -1).reshape(-1, 1)
		v = self.weights[i] @ x
		y = self._activation(v)
		activations.append(y)
	x = np.insert(activations[-1], 0, -1).reshape(-1, 1)
	v = self.weights[-1] @ x
	activations.append(v)
	return activations
```
$y(n)=phi[w^T (n)x(n)]$
=== Backward
```python
def _backward(self, activations, y):
	grads_W = [None] * len(self.weights)

	delta = y - activations[-1]
	grads_W[-1] = np.outer(delta.reshape(-1), np.insert(activations[-2], 0, -1))

	for i in reversed(range(len(self.weights) - 1)):
		delta = self._activation_derive(activations[i + 1]) * (self.weights[i + 1][:, 1:].T @ delta)
		grads_W[i] = np.outer(delta.reshape(-1), np.insert(activations[i], 0, -1))

	return grads_W
```
$delta_j=e_j (n)phi'(v_j (n))$

$delta_j(n)=phi'(v_j(n))sum_k delta_k (n)w_(k j)(n)$
=== 調整鍵結值
```python
self.weights[i] += self.learning_rate * grads_W[i]
```
$w_(j i)=w_(j i)+Delta w_(j i)=w_(j i)+eta times delta_j (n) times y_i (n)$
== 模擬程式
=== 模擬車運動
```python
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
```
$x(t+1)=x(t)+cos[phi(t)+theta(t)]+sin[theta(t)]sin[phi(t)]$

$y(t+1)=y(t)+sin[phi(t)+theta(t)]-sin[theta(t)]cos[phi(t)]$

$phi(t+1)=phi(t)-arcsin[(2sin[theta(t)])/b]$
=== 判斷點是否在多邊形內
```python
def inArea(self, point: Point2D) -> bool:
	winding_number = 0
	for edge in self.edges:
		if edge.point1.y <= point.y < edge.point2.y and edge.isLeft(point) > 0:
			winding_number += 1
		elif edge.point2.y < point.y <= edge.point1.y and edge.isLeft(point) < 0:
			winding_number -= 1
	return winding_number != 0
```
=== 判斷點的射線與多邊形邊的距離
```python
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
```
= 分析
1. 因為訓練資料較少的緣故，單看預測數字與資料集感覺差距很大，但在實際模擬時效果卻很好
2. 一開始對特徵進行標準化，但因為ReLU函數對負數值會直接為0導致訓練失敗，因此改成使用Min-Max與Leaky-ReLU
3. 因為初始權重隨機與ReLU函數特性(神經元死亡)，有機率會訓練失敗，在使用Leaky-ReLU後有減緩問題出現頻率
