import torch
import torch.nn as nn
import torch.optim as optim

# XOR 訓練資料
X = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])

y = torch.tensor([
    [0.0],
    [1.0],
    [1.0],
    [0.0]
])

# 建立多層前饋神經網路
model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)

# BCEWithLogitsLoss 已包含 Sigmoid
loss_function = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)

for epoch in range(3000):
    # 前向傳播
    logits = model(X)
    loss = loss_function(logits, y)

    # 清除上一輪梯度
    optimizer.zero_grad()

    # 反向傳播：計算梯度
    loss.backward()

    # 更新權重
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss = {loss.item():.6f}")

# 預測
with torch.no_grad():
    probability = torch.sigmoid(model(X))
    prediction = (probability >= 0.5).int()

    print("機率：")
    print(probability)

    print("分類結果：")
    print(prediction)
