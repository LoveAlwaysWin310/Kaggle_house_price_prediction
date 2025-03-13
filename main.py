import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


# 定义神经网络模型
class HousePricePredict(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(HousePricePredict, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.fc_out = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.fc_out(x)
        return x


# 定义 RMSLE 损失函数
class RMSLELoss(nn.Module):
    def __init__(self):
        super(RMSLELoss, self).__init__()

    def forward(self, y_pred, y_true):
        log_pred = torch.log1p(y_pred.clamp(min=0))
        log_true = torch.log1p(y_true)
        loss = torch.sqrt(torch.mean((log_pred - log_true) ** 2))
        return loss


# 训练参数
input_dim = 287  # 输入维度
hidden_dim1 = 128
hidden_dim2 = 32
output_dim = 1  # 输出维度
epoch = 300  # 训练轮数
batch_size = 128
lr = 0.1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# 载入数据
train_data = pd.read_csv('data/train.csv').iloc[:, 1:-1]  # 提取特征
test_data = pd.read_csv('data/test.csv').iloc[:, 1:]  # 测试集特征
features = pd.concat([train_data, test_data], axis=0)

# 处理数值型特征
numeric_type_features = features.select_dtypes(include=['number']).columns  # 选择数值型列
features[numeric_type_features] = features[numeric_type_features].apply(lambda x: (x - x.mean()) / x.std())
features[numeric_type_features] = features[numeric_type_features].fillna(0)  # 缺失值填充 0

# 处理分类特征
Nonnumerical_type_features = features.select_dtypes(exclude=['number']).columns
features = pd.get_dummies(features, columns=Nonnumerical_type_features, dummy_na=True, drop_first=True)

# 确保所有数据都是数值类型
features = features.apply(pd.to_numeric, errors='coerce')
features = features.fillna(0)

# 将数据转换为 PyTorch Tensor
train_num = train_data.shape[0]
train_dataset = torch.tensor(features.iloc[:train_num, :].values.astype(float), dtype=torch.float)
train_label = torch.tensor(pd.read_csv('data/train.csv').iloc[:, -1].values.astype(float), dtype=torch.float).reshape(
    -1, 1)
test_dataset = torch.tensor(features.iloc[train_num:, :].values.astype(float), dtype=torch.float)

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(train_dataset, train_label, test_size=0.1, random_state=25)
Train_dataset = TensorDataset(X_train, y_train)
Val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(Train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(Val_dataset, batch_size=batch_size, shuffle=False)

# 初始化模型
model = HousePricePredict(input_dim, hidden_dim1, hidden_dim2, output_dim)
optimizer = torch.optim.Adam(model.parameters(), lr)
criterion = RMSLELoss()
model.to(device)

train_losses = []
val_losses = []

# 训练模型
for i in range(epoch):
    model.train()
    total_loss = 0
    for train_x, train_y in train_loader:
        train_x, train_y = train_x.to(device), train_y.to(device)
        optimizer.zero_grad()
        y_pred = model(train_x)
        loss = criterion(y_pred, train_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_x, val_y in val_loader:
            val_x, val_y = val_x.to(device), val_y.to(device)
            y_val_pred = model(val_x)
            val_loss += criterion(y_val_pred, val_y).item()

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch [{i + 1}/{epoch}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

# 绘制训练曲线
plt.figure(figsize=(8, 5))
plt.plot(range(1, epoch + 1), train_losses, label='Train Loss')
plt.plot(range(1, epoch + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('RMSLE Loss')
plt.title('Training vs. Validation Loss')
plt.legend()
plt.show()

# 预测测试集
model.eval()
with torch.no_grad():
    pred_test = model(test_dataset.to(device))
    pred_test = pred_test.cpu().numpy().flatten()

# 生成提交文件
submission = pd.DataFrame({
    'Id': pd.read_csv('data/test.csv')['Id'],
    'SalePrice': pred_test
})
submission.to_csv('outputs/submission.csv', index=False)
print("Submission file saved.")
