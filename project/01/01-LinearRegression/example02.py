import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


# 拟合非线性函数（正弦波加高斯噪声）

# 1. 生成非线性数据集 (正弦函数 + 高斯噪声)
def generate_dataset(num_samples=1000):
    np.random.seed(42)
    x = np.linspace(-10, 10, num_samples)
    # 非线性函数: 正弦波 + 高斯噪声
    y = np.sin(x) * 2 + np.cos(x * 1.5) * 1.5 + 0.5 * np.random.normal(size=num_samples)
    return x, y


# 2. 定义神经网络模型
class NonlinearRegressor(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.net(x)


# 3. 训练函数
def train_model(model, x_train, y_train, epochs=1000, lr=0.01, batch_size=64):
    # 转换数据为PyTorch张量
    x_tensor = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    # 创建数据集和数据加载器
    dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 添加学习率调度器 - 使用余弦退火
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    # 训练循环
    losses = []
    print("开始训练...")
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_x, batch_y in dataloader:
            # 前向传播
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # 更新学习率
        scheduler.step()

        # 计算平均epoch loss
        epoch_loss /= len(dataloader)
        losses.append(epoch_loss)

        # 每10个epoch打印一次
        if (epoch + 1) % 10 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.6f}, LR: {current_lr:.2e}')

    print("训练完成!")
    return losses


# 4. 推理函数
def predict(model, x):
    with torch.no_grad():
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
        predictions = model(x_tensor)
    return predictions.squeeze().numpy()


# 5. 可视化结果
def plot_results(x, y_true, y_pred):
    plt.figure(figsize=(12, 6))
    plt.scatter(x, y_true, alpha=0.5, label='真实数据', s=10)
    plt.plot(x, y_pred, 'r-', linewidth=2, label='模型预测')
    plt.title('非线性函数拟合结果')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()


# 主程序
if __name__ == "__main__":
    # 生成数据集
    x, y = generate_dataset(2500)

    # 创建模型实例
    model = NonlinearRegressor(hidden_size=256)

    # 训练模型
    losses = train_model(
        model,
        x, y,
        epochs=500,
        lr=1e-4,
        batch_size=128
    )

    # 训练损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('训练损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    plt.show()

    # 推理并可视化
    test_x = np.linspace(-10.5, 10.5, 500)
    pred_y = predict(model, test_x)

    # 可视化拟合结果
    plot_results(test_x, np.sin(test_x) * 2 + np.cos(test_x * 1.5) * 1.5, pred_y)