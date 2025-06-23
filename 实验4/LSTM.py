import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 生成二进制数据
def int_to_bin(x, length=8):
    # 将整数转换为指定长度的二进制数组
    return np.array([int(b) for b in format(x, f'0{length}b')])

def generate_data(n_samples=10000):
    # 生成用于训练和测试的二进制数据
    data = []
    for _ in range(n_samples):
        a = np.random.randint(0, 128)  # 随机生成一个整数 a 范围在 0 到 127
        b = np.random.randint(0, 128)  # 随机生成一个整数 b 范围在 0 到 127
        sum_val = a + b  # 计算 a 和 b 的和
        input_seq = np.hstack([int_to_bin(a), int_to_bin(b)])  # 将 a 和 b 转换为二进制并拼接为输入序列，长度为 16
        target_seq = int_to_bin(sum_val % 256)  # 取 a + b 的最低 8 位作为目标输出序列
        data.append((input_seq, target_seq))  # 将输入序列和目标序列添加到数据列表中
    return data

# 自定义 LSTM 模型
class CustomLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1):
        super(CustomLSTM, self).__init__()
        self.hidden_size = hidden_size
        # 输入门、遗忘门、输出门和候选记忆单元的权重和偏置
        self.W_i = nn.Parameter(torch.Tensor(input_size + hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))
        self.W_f = nn.Parameter(torch.Tensor(input_size + hidden_size, hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))
        self.W_o = nn.Parameter(torch.Tensor(input_size + hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))
        self.W_c = nn.Parameter(torch.Tensor(input_size + hidden_size, hidden_size))
        self.b_c = nn.Parameter(torch.Tensor(hidden_size))
        # 全连接层用于将隐藏状态映射到输出
        self.fc = nn.Linear(hidden_size, 8)
        self.sigmoid = nn.Sigmoid()  # 使用 Sigmoid 激活函数将输出映射到 [0, 1]
        self.init_weights()

    def init_weights(self):
        # 初始化模型权重
        for param in self.parameters():
            if param.data.dim() >= 2:
                nn.init.xavier_uniform_(param)  # 使用 Xavier 均匀分布初始化权重
            else:
                nn.init.zeros_(param)  # 偏置初始化为 0

    def forward(self, x):
        # 前向传播
        batch_size, seq_len, _ = x.size()  # 获取批量大小和序列长度
        h_t = torch.zeros(batch_size, self.hidden_size).to(x.device)  # 初始化隐藏状态
        c_t = torch.zeros(batch_size, self.hidden_size).to(x.device)  # 初始化记忆单元状态

        for t in range(seq_len):
            x_t = x[:, t, :]  # 取当前时间步的输入
            combined = torch.cat((x_t, h_t), dim=1)  # 将当前输入与隐藏状态拼接
            i_t = torch.sigmoid(torch.matmul(combined, self.W_i) + self.b_i)  # 输入门计算
            f_t = torch.sigmoid(torch.matmul(combined, self.W_f) + self.b_f)  # 遗忘门计算
            o_t = torch.sigmoid(torch.matmul(combined, self.W_o) + self.b_o)  # 输出门计算
            c_hat_t = torch.tanh(torch.matmul(combined, self.W_c) + self.b_c)  # 候选记忆单元计算
            c_t = f_t * c_t + i_t * c_hat_t  # 更新记忆单元状态
            h_t = o_t * torch.tanh(c_t)  # 更新隐藏状态

        out = self.fc(h_t)  # 使用全连接层将隐藏状态映射为输出
        out = self.sigmoid(out)  # 使用 Sigmoid 激活函数将输出映射到 [0, 1]
        return out

# 训练模型
def train_model(model, data, epochs=100, batch_size=64, lr=0.01):
    criterion = nn.BCELoss()  # 使用二进制交叉熵损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # 使用 Adam 优化器
    loss_history = []  # 记录损失值

    for epoch in range(epochs):
        np.random.shuffle(data)  # 每个 epoch 打乱数据
        total_loss = 0
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i+batch_size]  # 获取当前批次的数据
            inputs = torch.Tensor([x[0] for x in batch_data]).unsqueeze(-1)  # 输入是长度为 16 的序列，添加一个维度
            targets = torch.Tensor([x[1] for x in batch_data])  # 目标是长度为 8 的序列
            
            outputs = model(inputs)  # 前向传播计算输出
            loss = criterion(outputs, targets)  # 计算损失
            optimizer.zero_grad()  # 清除上一步的梯度信息
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新模型参数

            total_loss += loss.item()  # 累加损失值

        avg_loss = total_loss / len(data)  # 计算平均损失
        loss_history.append(avg_loss)  # 将平均损失添加到历史记录中

        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {avg_loss}')  # 每 10 个 epoch 打印一次损失
    
    # 可视化训练过程中的损失值
    plt.plot(range(epochs), loss_history, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.show()

# 主程序
if __name__ == "__main__":
    # 数据生成
    data = generate_data(10000)
    
    # 模型初始化
    model = CustomLSTM()
    
    # 训练模型
    train_model(model, data)
    # 保存模型
    torch.save(model.state_dict(), 'custom_lstm_model.pth')
    ## 评估模式
    # model.load_state_dict(torch.load('custom_lstm_model.pth'))
    # model.eval()  # 设置模型为评估模式
    # 测试模型并打印结果
    test_data = generate_data(5)  # 生成 5 组测试数据
    correct = 0
    for input_seq, target_seq in test_data:
        input_tensor = torch.Tensor(input_seq).unsqueeze(0).unsqueeze(-1)  # 将输入转换为张量，添加 batch 维度和输入维度
        output = model(input_tensor)  # 前向传播计算输出
        predicted = (output.squeeze().detach().numpy() > 0.5).astype(int)  # 将输出值阈值化为二进制值
        a = int(''.join(map(str, input_seq[:8])), 2)  # 解析输入的第一个数
        b = int(''.join(map(str, input_seq[8:])), 2)  # 解析输入的第二个数
        sum_val = int(''.join(map(str, target_seq)), 2)  # 解析目标输出
        predicted_sum = int(''.join(map(str, predicted)), 2)  # 解析预测输出
        print(f'a: {a}, b: {b}, Actual Sum: {sum_val}, Predicted Sum: {predicted_sum}')  # 打印测试结果
        if np.array_equal(predicted, target_seq):
            correct += 1  # 如果预测值与目标值相等，则计数加一
    print(f'Test Accuracy: {correct / len(test_data) * 100:.2f}%')  # 打印测试准确率