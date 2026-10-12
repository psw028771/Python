# Python 演算法與 AI 練習

Python 演算法、最佳化方法與機器學習的練習程式集。每個檔案都是可以獨立執行的範例，並附有中文註解。

## 檔案說明

### 圖論

| 檔案 | 內容 |
|---|---|
| `dijkstras.py` | Dijkstra 最短路徑演算法，計算起點到各節點的最低成本 |
| `dijkstras_algorithm.py` | 同上的另一種寫法 |

### 最佳化演算法

| 檔案 | 內容 |
|---|---|
| `hill_climbing.py` | 爬山演算法，求 `f(x) = -(x-3)² + 10` 的最大值 |
| `simulated.py` | 模擬退火，求 `f(x) = x² + 10·sin(x)` 的最小值 |
| `genetic.py` | 基因演算法，以 5 位元染色體求 `f(x) = x²` 的最大值 |
| `ant_colony.PY` | 蟻群演算法，求解 6 個城市的旅行推銷員問題（TSP） |

### 機器學習

| 檔案 | 內容 |
|---|---|
| `gradient.py` | 用 NumPy 手寫梯度下降，擬合線性迴歸 `y = wx + b` |
| `backward.py` | 用 PyTorch 訓練多層神經網路，學習 XOR 問題 |
| `mnist.py` | 用 PyTorch 訓練 CNN 辨識 MNIST 手寫數字，並將模型存為 `mnist_cnn.pth` |

## 環境需求

- Python 3.8 以上
- 只用到標準函式庫的程式（`dijkstras*.py`、`hill_climbing.py`、`simulated.py`、`genetic.py`、`ant_colony.PY`）不需安裝任何套件
- `gradient.py` 需要 NumPy
- `backward.py`、`mnist.py` 需要 PyTorch；`mnist.py` 另需 torchvision

```bash
pip install numpy torch torchvision
```

## 執行方式

```bash
python dijkstras.py
python mnist.py
```

`mnist.py` 第一次執行時會自動下載 MNIST 資料集到 `./data` 資料夾。程式會自動選用 CUDA、MPS 或 CPU 執行。
