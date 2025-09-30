#import "@preview/cheq:0.3.0": checklist
#show: checklist.with(fill: luma(95%), stroke: blue, radius: .2em)
#set heading(numbering: "一.1.a")

#let assets = "doc_assets"

= 加分題完成功能
- [x] 三維資料圖形顯示介面
- [x] 能夠處理多維資料(四維以上)
- [!] 數字辨識(無自訂測資實作)
- [x] 可辨識兩群以上的資料

= 程式執行說明 (GUI功能說明)
#image(assets + "/gui.png")
- 左側顯示可調整參數
	- 選擇Data(選中後在右側顯示檔名)
	- 學習率
	- 執行Epoch數
	- 準確率限制(達到設定值後停止訓練)
	- 選擇是否使用多層感知機
		- 設定隱藏層大小
- 右側顯示訓練結果
	- 訓練的Epoch數
	- Test資料的準確率
	- 鍵結值
- 右下方為訓練按鈕, 若為2或3維資料, 會將結果圖彈出顯示
= 程式碼簡介
== 單層感知機
```python
for row in train_data:
	input = np.array([-1, *row[:-1]])
	if weight @ input < 0 and row[-1] == 1:
		weight = weight + learning_rate * input
	elif weight @ input > 0 and row[-1] == 0:
		weight = weight - learning_rate * input
```
計算網路輸出值:
$y(n)=phi[w^T (n)x(n)]$

$phi(v)=cases(+1 "if" v>=0, -1 "if" v<0)$

調整鍵結值向量: $w(n+1)=cases(w(n) "if分類正確", w(n) + eta x(n) "if" y(n) < 0, w(n) - eta x(n) "if" y(n) >= 0)$
== 多層感知機
=== 前饋
```python
y = [row[:-1]]
	for weight in weights:
		y.append(lib.sigmoid(weight @ [-1, *y[-1]]))
```
計算網路輸出值:
$y(n)=phi[w^T (n)x(n)]$

$phi(v)=cases(+1 "if" v>=0, -1 "if" v<0)$
=== 倒傳遞
```python
def delta_final(prediction, output):
	return (output - prediction) * prediction * (1 - prediction)
```
$delta_j=e_j (n)phi'(v_j (n))=(d_j(n)-O_j (n))O_j(n)(1-O_j (n))$
```python
for layer in range(len(layer_size)-2, 0, -1):
	delta = [y[layer] * (1 - y[layer]) * (weights[layer].T[1:] @ delta[0]), *delta]
```
$delta_j(n)=phi'(v_j(n))sum_k delta_k (n)w_(k j)(n)=y_j (n)(1-y_j (n))sum_k delta_k(n)w_(k j)(n)$
=== 調整鍵結值
```python
for layer in range(len(weights)):
	weights[layer] += learning_rate * np.outer(delta[layer], [-1, *y[layer]])
```
$w_(j i)=w_(j i)+Delta w_(j i)=w_(j i)+eta times delta_j (n) times y_i (n)$
=== 辨識多群資料
方法: 將輸出層從單個節點調至群數個, 並將標籤轉換成One-Hot(只有將標籤作為索引為1, 其餘為0的陣列), 最後預測輸出的陣列最高值的索引即是預測結果
= 實驗結果
所有訓練參數相同
- 學習率: 0.4
- Epoch: 500
- 準確率限制: 0.99
- 隱藏層大小: [20, 12, 6] (多層感知機)
== 基本題
#let slp_files = (
	"2cring",
	"2Ccircle1",
	"2Circle1",
	"2Circle2",
	"2CloseS",
	"2CloseS2",
	"2CloseS3",
	"2CS",
	"2Hcircle1",
	"2ring",
	"perceptron1",
	"perceptron2"
)

#grid(
	gutter: 2em,
	..slp_files.map(file => grid(
		columns: 2,
		gutter: 2em,
		block(align(center)[
			#image(assets + "/" + file + ".png", width: 90%)
			#text(0.8em, file)
		]),
		block(align(center)[
			#image(assets + "/" + file + "-mlp.png", width: 90%)
			#text(0.8em, file + " (mlp)")
		])
	))
)
== 加分題
#let mlp_files = (
	"5CloseS1",
	"C3D",
	"perceptron3",
	"perceptron4",
	"xor"
)

#grid(
	gutter: 2em,
	..mlp_files.map(file => grid(
		columns: 2,
		gutter: 2em,
		block(align(center)[
			#image(assets + "/" + file + ".png", width: 90%)
			#text(0.8em, file)
		]),
		block(align(center)[
			#image(assets + "/" + file + "-mlp.png", width: 90%)
			#text(0.8em, file + " (mlp)")
		])
	))
)
= 實驗結果分析及討論
+ 訓練次數過小容易造成未擬和, 訓練次數過大容易造成過擬和
	- 就結果而論應該選Test準確率最高的情況
	- 就實際考量(初始值為隨機情況下), 在Train Data的準確率上升變得非常緩慢時就可以考慮停止訓練
+ 多層感知機
	- 隱藏層設成2, 3時效果比只有一層好很多, 超過3層提升不明顯且時間花費巨大
	- 隱藏層越往輸出靠近每層神經元數目漸漸減少似乎有較好效果
