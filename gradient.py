import numpy as np

# 訓練資料
x = np.array([0, 1, 2, 3, 4], dtype=float)
y = np.array([1, 3, 5, 7, 9], dtype=float)

# 初始化模型參數
w = 0.0
b = 0.0

learning_rate = 0.01
epochs = 1000
n = len(x)

for epoch in range(epochs):
    # Forward pass：產生預測
    y_pred = w * x + b

    # 計算誤差
    error = y_pred - y

    # 計算梯度
    dw = (2 / n) * np.sum(error * x)
    db = (2 / n) * np.sum(error)

    # 更新參數
    w = w - learning_rate * dw
    b = b - learning_rate * db

    if (epoch + 1) % 100 == 0:
        loss = np.mean(error ** 2)
        print(
            f"Epoch {epoch + 1}: "
            f"loss={loss:.6f}, w={w:.4f}, b={b:.4f}"
        )

print(f"學習結果：y = {w:.4f}x + {b:.4f}")
