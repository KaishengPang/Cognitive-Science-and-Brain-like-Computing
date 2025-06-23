import numpy as np
import matplotlib.pyplot as plt
# 激活函数sigmoid：将输入映射到(0,1)区间
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# sigmoid函数的导数：用于反向传播过程中更新权重
def sigmoid_derivative(x):
    return x * (1 - x)

# 初始化网络的权重和偏置
def initialize_network(input_size, hidden_size, output_size):

    # 初始化输入层到隐藏层的权重，范围在[-1, 1]之间
    W_input_hidden = np.random.uniform(-1, 1, (input_size, hidden_size))

    # 初始化隐藏层到输出层的权重，范围在[-1, 1]之间
    W_hidden_output = np.random.uniform(-1, 1, (hidden_size, output_size))

    # 初始化隐藏层和输出层的偏置，范围在[-1, 1]之间
    b_hidden = np.random.uniform(-1, 1, (1, hidden_size))
    b_output = np.random.uniform(-1, 1, (1, output_size))

    # 返回初始化后的权重和偏置
    return W_input_hidden, W_hidden_output, b_hidden, b_output

# 前向传播：从输入层传到隐藏层，再到输出层
def forward_propagation(X, W_input_hidden, W_hidden_output, b_hidden, b_output):
    # 计算隐藏层的输入：输入乘以权重加上偏置
    hidden_input = np.dot(X, W_input_hidden) + b_hidden

    # 通过激活函数（sigmoid）计算隐藏层的输出
    hidden_output = sigmoid(hidden_input)

    # 计算输出层的输入：隐藏层输出乘以权重加上偏置
    final_input = np.dot(hidden_output, W_hidden_output) + b_output

    # 通过激活函数（sigmoid）计算最终输出
    final_output = sigmoid(final_input)

    # 返回隐藏层输出和最终输出
    return hidden_output, final_output

# 计算平方和误差（SSE）
def compute_loss(y_true, y_pred):
    return 0.5 * np.sum((y_true - y_pred) ** 2)

# 反向传播：通过计算误差来调整权重和偏置
def backward_propagation(X, y, hidden_output, final_output, W_hidden_output, W_input_hidden, b_output, b_hidden,
                         learning_rate=0.5):
    # 输出层误差：真实输出与预测输出之间的差距
    output_error = y - final_output

    # 输出层的delta值：误差乘以激活函数的导数，用于更新权重
    output_delta = output_error * sigmoid_derivative(final_output)

    # 隐藏层误差：通过反向传播计算输出层误差对隐藏层的影响
    hidden_error = output_delta.dot(W_hidden_output.T)

    # 隐藏层的delta值：误差乘以激活函数的导数，用于更新权重
    hidden_delta = hidden_error * sigmoid_derivative(hidden_output)

    # 更新隐藏层到输出层的权重
    W_hidden_output += hidden_output.T.dot(output_delta) * learning_rate

    # 更新输出层的偏置
    b_output += np.sum(output_delta, axis=0) * learning_rate

    # 更新输入层到隐藏层的权重
    W_input_hidden += X.T.dot(hidden_delta) * learning_rate

    # 更新隐藏层的偏置
    b_hidden += np.sum(hidden_delta, axis=0) * learning_rate

    # 返回更新后的权重和偏置
    return W_input_hidden, W_hidden_output, b_hidden, b_output

# 训练神经网络
def train(X, y, input_size, hidden_size, output_size, epochs=10000, learning_rate=0.5):
    # 初始化网络的权重和偏置
    W_input_hidden, W_hidden_output, b_hidden, b_output = initialize_network(input_size, hidden_size, output_size)
    # 用于记录每个epoch的损失值
    losses = []
    # 进行多个epoch的训练
    for epoch in range(epochs):
        # 前向传播：计算隐藏层和输出层的输出
        hidden_output, final_output = forward_propagation(X, W_input_hidden, W_hidden_output, b_hidden, b_output)

        # 计算当前epoch的损失值
        loss = compute_loss(y, final_output)
        losses.append(loss)  # 记录每个epoch的损失值

        # 反向传播：根据损失更新权重和偏置
        W_input_hidden, W_hidden_output, b_hidden, b_output = backward_propagation(X, y, hidden_output, final_output,
                                                                                   W_hidden_output, W_input_hidden,
                                                                                   b_output, b_hidden, learning_rate)

        # 每10个epoch打印一次当前损失
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")

        # 如果损失小于预设值（1e-5），则停止训练
        if loss < 1e-5:
            print(f"Training stopped at epoch {epoch}")
            break
            # 绘制损失函数变化曲线
    plt.plot(losses)
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    # 返回训练好的权重和偏置
    return W_input_hidden, W_hidden_output, b_hidden, b_output


X = np.array([[0.35, 0.9]])  
y = np.array([[0.5]])  

# 定义神经网络的参数
input_size = 2  # 输入层的神经元个数（2个输入）
hidden_size = 2  # 隐藏层的神经元个数（2个神经元）
output_size = 1  # 输出层的神经元个数（1个输出）
learning_rate = 0.5  # 学习率：控制权重更新的幅度
epochs = 10000  # 训练轮数：最大训练次数

# 训练神经网络
W_input_hidden, W_hidden_output, b_hidden, b_output = train(X, y, input_size, hidden_size, output_size, epochs,
                                                            learning_rate)

# 测试训练后的模型
hidden_output, final_output = forward_propagation(X, W_input_hidden, W_hidden_output, b_hidden, b_output)

# 打印训练后模型的最终输出
print("Final output after training:")
print(final_output)