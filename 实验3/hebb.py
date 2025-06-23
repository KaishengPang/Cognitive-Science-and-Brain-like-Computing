import numpy as np
import matplotlib.pyplot as plt

# 定义Hebb学习规则函数
def hebb_learning_adjusted(training_data, learning_rate=0.1):
    """
    使用Hebb学习规则训练网络。
    
    参数:
    training_data: list，训练数据，包含多个数字矩阵。
    learning_rate: float，学习率，控制权重更新的步幅。
    
    返回值:
    weights: numpy array，训练后的权重矩阵。
    """
    # 获取训练数据中每个数字的大小(行数×列数)，即输入向量的维度
    input_size = training_data[0].shape[0] * training_data[0].shape[1]
    
    # 初始化权重矩阵为小的随机值（大小为input_size×input_size）
    weights = np.random.randn(input_size, input_size) * 0.01
    
    # 遍历每个训练数据，应用Hebb学习规则
    for data in training_data:
        # 将二维矩阵展平为一维向量
        vectorized_input = np.array(data).flatten()
        # 使用Hebb学习规则更新权重矩阵: ΔWij = η * xj * yi
        weights += learning_rate * np.outer(vectorized_input, vectorized_input)

    return weights

# 定义测试网络的函数
def test_hebb(weights, test_data):
    """
    测试Hebb网络的函数。
    
    参数:
    weights: numpy array，训练好的权重矩阵。
    test_data: numpy matrix，测试数据矩阵。
    
    返回值:
    output_binarized: numpy array，二值化后的网络输出。
    """
    # 将测试数据展平为向量
    test_vector = np.array(test_data).flatten()
    test_vector = test_vector.reshape(-1, 1)  # 将一维向量变为列向量
    
    # 计算权重矩阵和测试向量的乘积，得到网络的输出
    output = np.dot(weights, test_vector).flatten()
    
    # 对输出进行二值化处理：大于0的变为1，小于0的变为-1
    output_binarized = np.sign(output)

    # return output
    return output_binarized

# 定义数字图案，每个数字是一个6x5的矩阵
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
                    [1, 1, 1, 1, 1]])
}

# 创建训练数据列表，包含无噪声的数字图案
training_data = [digits['0'], digits['1'], digits['2']]

# 训练Hebb网络，使用上述定义的训练数据
weights = hebb_learning_adjusted(training_data)

# 创建带噪声的测试数据，数字"0"的图案上添加一些随机噪声
# test_data = np.where(np.random.rand(6, 5) < 0.4, -digits['2'], digits['2'])
test_data = digits['2'] + np.random.randn(6, 5) * 3

# 使用训练后的权重对测试数据进行预测
output = test_hebb(weights, test_data)

# 定义一个函数，用于可视化原始图像、加噪声图像和识别结果
def plot_images(original, noisy, recognized):
    """
    绘制三个图片：原始图片、加噪声后的图片、以及识别出的图片。
    
    参数:
    original: numpy matrix，原始图像。
    noisy: numpy matrix，噪声图像。
    recognized: numpy matrix，识别出的图像。
    """
    plt.figure(figsize=(12, 4))  # 调整图像窗口大小
    
    plt.subplot(1, 3, 1)  # 创建第一个子图：显示原始图像
    plt.imshow(original, cmap='gray_r')
    plt.title('Original')  # 设置标题
    plt.axis('off')  # 关闭坐标轴
    
    plt.subplot(1, 3, 2)  # 创建第二个子图：显示加噪声后的图像
    plt.imshow(noisy, cmap='gray_r')
    plt.title('Noised')  # 设置标题
    plt.axis('off')  # 关闭坐标轴
    
    plt.subplot(1, 3, 3)  # 创建第三个子图：显示识别出的图像
    plt.imshow(recognized, cmap='gray_r')
    plt.title('Recognized')  # 设置标题
    plt.axis('off')  # 关闭坐标轴
    
    plt.show()  # 显示图像窗口

# 将测试输出重新调整为6x5的矩阵，并调用可视化函数显示结果
output_image = output.reshape(6, 5)
plot_images(digits['2'], test_data, output_image)