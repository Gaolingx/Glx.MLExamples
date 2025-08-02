import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 二分类问题（具体为 OR 门逻辑：x1 OR x2 = y）

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self):
        # 定义数据集：x1, x2, y
        self.data = torch.tensor([
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],  # 补充一个样本使训练更稳定
        ], dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.data[idx, :2]  # 前两列是x1, x2
        target = self.data[idx, 2]  # 第三列是y
        return features, target


# 定义线性回归模型
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(2, 1)  # 输入特征2个，输出1个
        self.sigmoid = nn.Sigmoid()  # 将输出映射到[0,1]范围

    def forward(self, x):
        out = self.linear(x)
        return self.sigmoid(out)


# 训练函数
def train_model():
    # 设置随机种子确保可重复性
    torch.manual_seed(42)

    # 初始化数据集和数据加载器
    dataset = CustomDataset()
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 初始化模型、损失函数和优化器
    model = LinearRegressionModel()
    criterion = nn.MSELoss()  # 均方误差损失
    optimizer = optim.SGD(model.parameters(), lr=5e-2)

    # 训练参数
    num_epochs = 500

    # 训练循环
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for inputs, targets in dataloader:
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # 每个epoch输出平均损失
        if (epoch + 1) % 10 == 0:
            print(f'Train: Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(dataloader):.6f}')

    print(f'Train: Finished. Total Epoch: {num_epochs}')
    print()

    return model


# 推理函数
def inference(model, test_cases):
    model.eval()
    with torch.no_grad():
        for i, (x1, x2, expected) in enumerate(test_cases):
            inputs = torch.tensor([[x1, x2]], dtype=torch.float32)
            prediction = model(inputs).item()
            print(f'Group {i + 1}:')
            print(f'输入 {int(x1)} {int(x2)}')
            print(f'预测输出: {prediction:.4f} → 四舍五入: {round(prediction)}')
            print(f'期望输出: {expected}')
            print()


if __name__ == "__main__":
    # 训练模型
    trained_model = train_model()

    # 定义测试用例
    test_data = [
        (1, 0, 1),
        (0, 0, 0),
        (1, 1, 1)
    ]

    # 进行推理
    inference(trained_model, test_data)