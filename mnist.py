import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# 1. 設置運算裝置 (GPU / MPS / CPU)
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")

# 2. 資料前處理與載入
# MNIST 資料為 28x28 灰階影像，標準化轉換使用 MNIST 全局均值 0.1307 與標準差 0.3081
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(
    dataset=train_dataset, batch_size=64, shuffle=True, num_workers=2
)
test_loader = DataLoader(
    dataset=test_dataset, batch_size=1000, shuffle=False, num_workers=2
)


# 3. 定義 CNN 模型架構
class ConvNet(nn.Module):

  def __init__(self):
    super().__init__()
    # 輸入通道 1，輸出通道 32，卷積核 3x3
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
    # 輸入通道 32，輸出通道 64，卷積核 3x3
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    self.dropout1 = nn.Dropout(0.25)
    self.dropout2 = nn.Dropout(0.5)
    # 28x28 -> conv1(26x26) -> conv2(24x24) -> pool(12x12) -> 64 * 12 * 12
    self.fc1 = nn.Linear(64 * 12 * 12, 128)
    self.fc2 = nn.Linear(128, 10)  # 10 個類別 (0-9)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = self.pool(x)
    x = self.dropout1(x)
    x = torch.flatten(x, 1)  # 展平特徵圖
    x = F.relu(self.fc1(x))
    x = self.dropout2(x)
    x = self.fc2(x)
    return x


# 4. 初始化模型、損失函數與優化器
model = ConvNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# 5. 訓練與評估函數
def train_epoch(model, dataloader, criterion, optimizer, device):
  model.train()
  running_loss = 0.0
  correct = 0
  total = 0

  for data, target in dataloader:
    data, target = data.to(device), target.to(device)

    optimizer.zero_grad()  # 清空前一步梯度
    output = model(data)  # 前向傳播
    loss = criterion(output, target)  # 計算損失
    loss.backward()  # 反向傳播計算梯度
    optimizer.step()  # 更新參數

    running_loss += loss.item() * data.size(0)
    pred = output.argmax(dim=1)
    correct += (pred == target).sum().item()
    total += data.size(0)

  epoch_loss = running_loss / total
  epoch_acc = 100.0 * correct / total
  return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
  model.eval()
  test_loss = 0.0
  correct = 0
  total = 0

  with torch.no_grad():  # 評估階段關閉梯度計算以節省顯存與加速
    for data, target in dataloader:
      data, target = data.to(device), target.to(device)
      output = model(data)
      test_loss += criterion(output, target).item() * data.size(0)
      pred = output.argmax(dim=1)
      correct += (pred == target).sum().item()
      total += data.size(0)

  test_loss = test_loss / total
  test_acc = 100.0 * correct / total
  return test_loss, test_acc


# 6. 執行訓練循環 (3 個 Epoch 作為示範)
epochs = 3
for epoch in range(1, epochs + 1):
  train_loss, train_acc = train_epoch(
      model, train_loader, criterion, optimizer, device
  )
  test_loss, test_acc = evaluate(model, test_loader, criterion, device)

  print(
      f"Epoch {epoch:02d} | "
      f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% | "
      f"Test Loss: {test_loss:.4f} - Test Acc: {test_acc:.2f}%"
  )

# 7. 儲存模型權重
torch.save(model.state_dict(), "mnist_cnn.pth")
print("Model saved to mnist_cnn.pth")