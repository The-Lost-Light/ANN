#import "@preview/cheq:0.3.0": checklist
#show: checklist.with(fill: luma(95%), stroke: blue, radius: .2em)
#set heading(numbering: "一.1.a")

#let assets = "assets"

= Hopfield Network

= 加分題完成功能
- [x] 使用Bonus_Training.txt 和 Bonus_Testing.txt資料集
- [x] 可選擇同步或非同步更新
- [x] 可選擇是否使用Theta(=0)

= 程式執行說明 (GUI功能說明)
#image(assets + "/gui.png")
- 左上角可以選擇Basic或Bonus資料集
- 左下角可一選擇pattern
- 中間顯示結果(train, test, 及四種結果)

= Hopfield 程式碼簡介
```python
def train(self, patterns, threadhold=False):
	N = len(patterns)
	if N == 0:
		return

	for p in patterns:
		self.weights += np.outer(p, p.T)

	self.weights -= N * np.identity(self.pattern_size)
	self.weights /= self.pattern_size
	self.theta = np.sum(self.weights, axis=0).reshape(-1, 1) if threadhold else np.zeros((len(self.weights), 1))
```
$W=1/p sum_(k=1)^N x_k x_k^T-N/P I$

$theta_j=sum_(i=1)^p w_(j i)$ 或是 $theta=0$

```python
def _update_unit(self, x, i=None):
	if i is not None:
		v = self.weights[i] @ x - self.theta[i]
	else:
		v = self.weights @ x - self.theta
	for j in range(len(v)):
		if v[j] > 0:
			v[j] = 1
		elif v[j] == 0:
			v[j] = x[j]
		elif v[j] < 0:
			v[j] = -1
	return v

def predict(self, pattern, max_iter=1000, asynchronous=False):
	x = pattern.copy()
	for iter in range(max_iter):
		old_x = x.copy()
		if asynchronous:
			i = np.random.randint(0, self.pattern_size)
			x[i] = self._update_unit(x, i)[0]
		else:
			x = self._update_unit(x)
		if np.array_equal(x, old_x) and iter >= 100:
			break
	return x
```
將是否同步分開處理，若為非同步每次更新一個row

= 實驗結果

== Basic
#grid(
  gutter: 2em,
  ..range(1, 4).map(n => {
    let filename = "assets/basic_pattern" + str(n) + ".png"

    block(align(center)[
      #figure(
        image(filename, width: 100%),
        caption: [Basic #n],
        numbering: none
      )
    ])
  })
)

== 加分題
#grid(
  gutter: 2em,
  ..range(1, 16).map(n => {
    let filename = "assets/bonus_pattern" + str(n) + ".png"

    block(align(center)[
      #figure(
        image(filename, width: 100%),
        caption: [Bonus #n],
        numbering: none
      )
    ])
  })
)

= 實驗結果分析及討論
- 同步在記憶數量少的情況下幾乎可以完美回憶, 但可能會出現Bonus Pattern 6這種完全回憶錯誤的情況
- 非同步在大部分情況圖形都會有缺角破損的情況，但是卻沒有明顯圖形出錯的情況，在較多需要記憶的內容看起來效果更好
- Theta=0看起來在絕大多數情況都比較差(在Basic時兩者完全沒差距)，但在Bonus且非同步時有少數效果更好，可能是要記憶的內容
