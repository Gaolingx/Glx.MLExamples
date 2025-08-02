import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 拟合函数f(x)=2x+1

# 1. 创建数据集 (满足 f(x)=2x+1 并添加噪声)
def generate_data(samples=200):
    np.random.seed(42)
    x = np.linspace(-5, 5, samples)
    y = 2 * x + 1 + np.random.normal(0, 1.0, samples)  # 添加高斯噪声
    return x.astype(np.float32), y.astype(np.float32)


# 转换为PyTorch张量
x_data, y_data = generate_data()
x_tensor = torch.from_numpy(x_data).unsqueeze(1)  # 转换为200x1张量
y_tensor = torch.from_numpy(y_data).unsqueeze(1)


# 2. 定义神经网络模型
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.layer1 = nn.Linear(1, 10)  # 输入1维，输出10维
        self.layer2 = nn.Linear(10, 1)  # 输出1维
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        return self.layer2(x)


model = LinearModel()

# 3. 训练配置
criterion = nn.MSELoss()  # 均方误差损失
optimizer = optim.Adam(model.parameters(), lr=5e-3)  # Adam优化器
epochs = 500

# 存储训练损失
loss_history = []

# 4. 训练循环
print("开始训练...")
for epoch in range(epochs):
    # 前向传播
    outputs = model(x_tensor)
    loss = criterion(outputs, y_tensor)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 记录损失
    loss_history.append(loss.item())

    # 每50个epoch打印一次损失
    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}')

print("训练完成！")

# 5. 绘制训练损失曲线
plt.plot(loss_history)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.grid(True)
plt.show()


# 6. 推理验证
def predict(x):
    model.eval()  # 切换到评估模式
    with torch.no_grad():
        x_tensor = torch.tensor([[x]], dtype=torch.float32)
        prediction = model(x_tensor)
    return prediction.item()


# 测试一些示例值
test_values = [-3, 0, 2.5, 4.7]
print("\n推理测试:")
for x in test_values:
    print(f'输入 x = {x:.1f}, 预测 y = {predict(x):.2f}, 期望 y = {2 * x + 1:.2f}')

# 7. 可视化拟合效果
plt.scatter(x_data, y_data, label='原始数据', alpha=0.6)
x_test = np.linspace(-5, 5, 100)
y_pred = [predict(x) for x in x_test]
plt.plot(x_test, y_pred, 'r-', linewidth=2, label='模型预测')
plt.title('神经网络拟合效果')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()