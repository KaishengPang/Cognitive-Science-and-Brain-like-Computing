import numpy as np  # 导入NumPy库，用于数值计算

# 定义数字0-9的点阵表示
digits = {
    '0': np.matrix([[1, 1, 1, 1, 1],
                    [1, -1, -1, -1, 1],
                    [1, -1, -1, -1, 1],
                    [1, -1, -1, -1, 1],
                    [1, -1, -1, -1, 1],
                    [1, 1, 1, 1, 1]]),

    '1': np.matrix([[-1, -1, 1, -1, -1],
                    [-1, -1, 1, -1, -1],
                    [-1, -1, 1, -1, -1],
                    [-1, -1, 1, -1, -1],
                    [-1, -1, 1, -1, -1],
                    [-1, -1, 1, -1, -1]]),

    '2': np.matrix([[1, 1, 1, -1, -1],
                    [-1, -1, -1, 1, -1],
                    [-1, -1, -1, 1, -1],
                    [-1, 1, 1, -1, -1],
                    [1, -1, -1, -1, -1],
                    [1, 1, 1, 1, 1]]),

    '3': np.matrix([[1, 1, 1, 1, 1],
                    [-1, -1, -1, -1, 1],
                    [1, 1, 1, 1, -1],
                    [1, 1, 1, 1, -1],
                    [-1, -1, -1, -1, 1],
                    [1, 1, 1, 1, 1]]),

    '4': np.matrix([[-1, -1, 1, 1, -1],
                    [-1, 1, -1, 1, -1],
                    [1, -1, -1, 1, -1],
                    [1, 1, 1, 1, 1],
                    [-1, -1, -1, 1, -1],
                    [-1, -1, -1, 1, -1]]),

    '5': np.matrix([[1, 1, 1, 1, 1],
                    [1, -1, -1, -1, -1],
                    [-1, 1, -1, -1, -1],
                    [-1, -1, 1, 1, 1],
                    [-1, -1, -1, -1, 1],
                    [1, 1, 1, 1, 1]]),

    '6': np.matrix([[-1, 1, 1, 1, 1],
                    [-1, 1, -1, -1, -1],
                    [1, 1, 1, 1, 1],
                    [1, -1, -1, -1, 1],
                    [1, -1, -1, -1, 1],
                    [1, 1, 1, 1, 1]]),

    '7': np.matrix([[1, 1, 1, 1, 1],
                    [-1, -1, -1, -1, 1],
                    [-1, -1, -1, 1, -1],
                    [-1, -1, 1, -1, -1],
                    [-1, 1, -1, -1, -1],
                    [1, -1, -1, -1, -1]]),

    '8': np.matrix([[1, 1, 1, 1, 1],
                    [1, -1, -1, -1, 1],
                    [-1, 1, 1, 1, -1],
                    [-1, 1, 1, 1, -1],
                    [1, -1, -1, -1, 1],
                    [1, 1, 1, 1, 1]]),

    '9': np.matrix([[1, 1, 1, 1, 1],
                    [1, -1, -1, -1, 1],
                    [1, -1, -1, -1, 1],
                    [1, 1, 1, 1, 1],
                    [-1, -1, -1, 1, -1],
                    [1, 1, 1, -1, -1]])
}

# 将数字点阵展平并存储为目标向量
target_vectors = np.array([digit.flatten().A1 for digit in digits.values()])

# 定义Hopfield神经网络类
class HopfieldNetwork:
    def __init__(self, size):
        self.size = size  # 网络中节点的数量
        self.weights = np.zeros((size, size))  # 初始化权重矩阵为零

    def train(self, patterns):
        # 根据输入的模式训练权重矩阵
        for p in patterns:
            self.weights += np.outer(p, p)  # 计算外积并累加到权重矩阵
        np.fill_diagonal(self.weights, 0)  # 将对角线置为零，避免自连接

    def predict(self, pattern, steps=100):
        # 对输入的模式进行预测
        pattern = pattern.copy()  # 创建输入模式的副本
        for _ in range(steps):
            pattern = np.sign(np.dot(self.weights, pattern))  # 更新模式
        return pattern  # 返回最终的预测结果

# 创建Hopfield网络并训练
network = HopfieldNetwork(size=30)  # 初始化网络，大小为30
network.train(target_vectors)  # 用目标向量训练网络

def add_noise(pattern, noise_level=0.1):
    # 添加噪声到输入模式
    noisy_pattern = pattern.copy()  # 创建输入模式的副本
    for i in range(len(noisy_pattern)):
        if np.random.rand() < noise_level:  # 根据噪声水平决定是否反转值
            noisy_pattern[i] = -noisy_pattern[i]  # 反转值
    return noisy_pattern  # 返回带噪声的模式

# 示例：产生带噪声的点阵
original_number = digits['6'].flatten().A1  # 选择数字展平
noisy_number = add_noise(original_number)  # 添加噪声

import matplotlib.pyplot as plt  # 导入Matplotlib库用于可视化

def visualize_combined(original, noisy, output):
    # 创建一个包含三个子图的可视化函数
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))  # 一行三列的子图

    # 原图
    axs[0].imshow(original.reshape(6, 5), cmap='gray_r')
    axs[0].set_title("Original")
    axs[0].axis('off')  # 不显示坐标轴

    # 带噪声的图像
    axs[1].imshow(noisy.reshape(6, 5), cmap='gray_r')
    axs[1].set_title("Noisy Input")
    axs[1].axis('off')  # 不显示坐标轴

    # 识别后的图像
    axs[2].imshow(output.reshape(6, 5), cmap='gray_r')
    axs[2].set_title("Network Output")
    axs[2].axis('off')  # 不显示坐标轴

    plt.tight_layout()  # 调整布局
    plt.show()  # 显示图形

# 识别带噪声的数字
output = network.predict(noisy_number)  # 输入带噪声的模式进行预测

# 可视化原图、带噪声的图像和识别后的图像
visualize_combined(original_number, noisy_number, output)
