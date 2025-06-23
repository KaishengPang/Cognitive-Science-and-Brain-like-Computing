import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 定义TopoICA模型
class TopoICA(nn.Module):
    def __init__(self, input_dim, output_dim, m=10):
        super(TopoICA, self).__init__()
        self.m = m
        # 多层感知机部分，用于非线性映射
        self.fc1 = nn.Linear(input_dim, 256)  # 输入层到隐藏层，输入维度为 input_dim，输出为 256
        self.fc2 = nn.Linear(256, 128)  # 隐藏层，输入为 256，输出为 128
        self.fc3 = nn.Linear(128, m)  # 输出独立成分 si，输出维度为 m
        self.fc_out = nn.Linear(m, output_dim)  # 最后的分类层，将独立成分映射到输出类别
        
        # 将 a_weights 的形状改为 [m, m]，使其与 s 匹配
        self.a_weights = nn.Parameter(torch.randn(m, m))  # 权重参数 a_weights，用于计算 I_hat
        self.activation = nn.ReLU()  # 使用 ReLU 激活函数

    def forward(self, x):
        x = torch.flatten(x, 1)  # 展平输入，保证输入维度为 [batch_size, input_dim]
        
        # 非线性激活
        x = self.activation(self.fc1(x))  # 输入经过第一层全连接，再经过 ReLU 激活
        x = self.activation(self.fc2(x))  # 经过第二层全连接和 ReLU 激活
        s = self.activation(self.fc3(x))  # 计算独立成分 s，形状为 [batch_size, m]
        
        # 输出层，分类为 0-9
        output = self.fc_out(s)  # 最终得到的分类结果
        
        # 计算 I_hat，即图像 I(x, y) 的线性组合
        I_hat = torch.matmul(s, self.a_weights)  # 计算 I_hat，形状应该与 s 匹配，即 [batch_size, m]
        
        # 检查数据格式，确保 I_hat 和 s 形状匹配
        assert I_hat.shape == s.shape, f"I_hat shape {I_hat.shape} does not match s shape {s.shape}"
        
        return output, I_hat

# 定义能量相关性损失
# 能量相关性损失用于约束独立成分之间的关系
# 通过最小化能量损失来减少成分之间的相互依赖性
def energy_correlation_loss(s):
    s_square = s ** 2  # 计算每个独立成分的平方
    cov_matrix = torch.cov(s_square.T)  # 计算平方后的协方差矩阵
    energy_loss = torch.mean(cov_matrix)  # 使用均值作为损失，鼓励协方差趋于 0
    return energy_loss

# 定义训练函数
def train(model, train_loader, criterion, optimizer, epoch):
    model.train()  # 设置模型为训练模式
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()  # 梯度清零
        output, I_hat = model(data)  # 得到模型的分类输出和线性组合
        
        # 分类损失
        classification_loss = criterion(output, target)  # 计算交叉熵损失
        
        # 能量相关性损失
        energy_loss = energy_correlation_loss(I_hat)  # 计算能量相关性损失
        
        # 总损失 = 分类损失 + 能量相关性损失
        loss = classification_loss + 0.1 * energy_loss  # 损失由分类损失和能量损失组成
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 更新权重

        # 每 100 个 batch 输出一次训练状态
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')

# 定义测试函数
def test(model, test_loader, criterion):
    model.eval()  # 设置模型为评估模式
    test_loss = 0  # 初始化测试损失
    correct = 0  # 初始化正确分类的数量
    with torch.no_grad():  # 在测试中不需要计算梯度
        for data, target in test_loader:
            output, _ = model(data)  # 只计算分类部分
            test_loss += criterion(output, target).item()  # 累加测试损失
            pred = output.argmax(dim=1, keepdim=True)  # 获取每个样本的预测类别
            correct += pred.eq(target.view_as(pred)).sum().item()  # 统计正确分类的数量

    test_loss /= len(test_loader.dataset)  # 计算平均损失
    accuracy = 100. * correct / len(test_loader.dataset)  # 计算准确率
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')

# MNIST 数据集加载和预处理
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])  # 图像归一化

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)  # 加载训练数据集
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)  # 加载测试数据集

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)  # 数据加载器，用于批量加载训练数据
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)  # 数据加载器，用于批量加载测试数据

# 实例化模型
input_dim = 28 * 28  # MNIST 图像大小为 28x28
output_dim = 10  # 10 个数字分类
model = TopoICA(input_dim=input_dim, output_dim=output_dim, m=10)  # 实例化 TopoICA 模型

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # 使用随机梯度下降优化器

# 训练和测试模型
for epoch in range(1, 11):  # 训练 10 个周期
    train(model, train_loader, criterion, optimizer, epoch)  # 训练模型
    test(model, test_loader, criterion)  # 测试模型

# 随机选取一些测试集样本进行可视化
# 用于可视化模型对测试集样本的预测结果
def visualize_predictions(model, test_loader):
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():
        data, target = next(iter(test_loader))  # 获取一批测试集样本
        output, _ = model(data)  # 获取预测结果
        pred = output.argmax(dim=1, keepdim=True).squeeze()  # 获取预测类别
        
        # 使用 plt.subplots 创建图像
        fig, axes = plt.subplots(1, 8, figsize=(15, 3))  # 一行 8 个子图
        for i in range(8):  # 只显示前 8 个样本
            axes[i].imshow(data[i].squeeze().cpu().numpy(), cmap='gray')  # 显示每个样本的灰度图像
            # 在子图标题中显示预测标签和真实标签
            axes[i].set_title(f'Pred: {pred[i].item()}\nTrue: {target[i].item()}')
            axes[i].axis('off')  # 关闭坐标轴显示

        # 显示图像并保持窗口
        plt.tight_layout()  # 自动调整子图布局，防止重叠
        plt.show()

# 可视化预测结果
visualize_predictions(model, test_loader)

# 可视化独立成分
# 用于可视化模型学到的独立成分
def visualize_independent_components(model, test_loader):
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():
        data, _ = next(iter(test_loader))  # 获取一批测试集样本
        _, I_hat = model(data)  # 获取估计的独立成分 I(x, y)
        
        # 可视化 I_hat 中的前 8 个样本
        fig, axes = plt.subplots(1, 8, figsize=(15, 3))
        for i in range(8):
            # 假设 I_hat 的维度是 [batch_size, m]，m 是独立成分数
            # 因此我们需要将 m 转换为一个矩阵形状来可视化
            component = I_hat[i].cpu().numpy().reshape(1, -1)  # 将每个成分展开为 1D 数组
            axes[i].imshow(component, cmap='gray', aspect='auto')  # 可视化每个独立成分
            axes[i].set_title(f'Component {i}')
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

# 调用函数进行独立成分可视化
visualize_independent_components(model, test_loader)

# 可视化协方差矩阵
# 用于可视化独立成分之间的相关性
def visualize_correlation_matrix(model, test_loader):
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():
        data, _ = next(iter(test_loader))  # 获取一批测试集样本
        _, I_hat = model(data)  # 获取估计的独立成分 I(x, y)
        
        # 计算每个成分的平方（假设独立成分之间的相关性与平方相关）
        I_square = I_hat.cpu().numpy() ** 2  # 计算 I_hat 的平方
        
        # 计算协方差矩阵
        cov_matrix = np.cov(I_square.T)  # I_square 转置，使得每个成分为一列
        
        # 可视化协方差矩阵
        plt.figure(figsize=(10, 8))
        sns.heatmap(cov_matrix, annot=True, fmt=".2f", cmap='coolwarm')  # 使用热力图显示协方差矩阵
        plt.title('Covariance Matrix of Independent Components')
        plt.show()

# 调用函数可视化协方差矩阵
visualize_correlation_matrix(model, test_loader)
