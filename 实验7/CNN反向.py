import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 卷积操作函数
def convolution(input_img, kernel, stride, padding):
    """
    输入:
        input_img: 2D 输入图像 (numpy array)
        kernel: 2D 卷积核 (numpy array)
        stride: 步幅 (int)
        padding: 填充大小 (int)
    输出:
        output: 经过卷积后的 2D 图像 (numpy array)
    功能:
        对输入图像执行卷积操作，返回卷积后的特征图。
    """
    k_h, k_w = kernel.shape  # 获取卷积核的高度和宽度
    # 在输入图像周围进行填充操作
    padded_img = np.pad(input_img, padding, mode='constant')  
    # 计算卷积后的输出图像的高度和宽度
    out_h = ((padded_img.shape[0] - k_h) // stride) + 1
    out_w = ((padded_img.shape[1] - k_w) // stride) + 1
    output = np.zeros((out_h, out_w))  # 初始化输出图像
    # 执行卷积操作
    for row in range(out_h):
        for col in range(out_w):
            # 计算对应窗口下的卷积值
            output[row, col] = np.sum(kernel * padded_img[row * stride:row * stride + k_h, col * stride:col * stride + k_w])
    return output

# ReLU 激活函数
def relu(matrix):
    """
    输入:
        matrix: 2D 或 1D 输入矩阵 (numpy array)
    输出:
        返回经过 ReLU 激活函数后的矩阵 (numpy array)
    功能:
        对输入矩阵中的每个元素应用 ReLU 激活函数 (max(0, x))。
    """
    return np.maximum(0, matrix)

# 最大池化操作
def max_pool(image, pool_size, step):
    """
    输入:
        image: 2D 输入图像 (numpy array)
        pool_size: 池化窗口大小 (tuple)
        step: 池化步幅 (int)
    输出:
        pooled_result: 经过最大池化后的 2D 图像 (numpy array)
    功能:
        对输入图像执行最大池化操作，返回池化后的特征图。
    """
    p_h, p_w = pool_size  # 获取池化窗口的高度和宽度
    out_h = (image.shape[0] - p_h) // step + 1  # 池化后图像的高度
    out_w = (image.shape[1] - p_w) // step + 1  # 池化后图像的宽度
    pooled_result = np.zeros((out_h, out_w))  # 初始化池化结果
    # 执行最大池化操作
    for row in range(out_h):
        for col in range(out_w):
            pooled_result[row, col] = np.max(image[row * step:row * step + p_h, col * step:col * step + p_w])
    return pooled_result

# 全连接层 (Dense layer)
def fully_connected(input_vec, w_matrix, b_vec):
    """
    输入:
        input_vec: 展平后的输入向量 (numpy array)
        w_matrix: 权重矩阵 (numpy array)
        b_vec: 偏置向量 (numpy array)
    输出:
        dense_out: 全连接层的输出 (numpy array)
    功能:
        计算全连接层的输出: y = Wx + b。
    """
    return np.dot(input_vec, w_matrix) + b_vec

# Softmax 函数
def softmax(vec):
    """
    输入:
        vec: 1D 输入向量 (numpy array)
    输出:
        softmax_out: 应用 softmax 后的向量 (numpy array)
    功能:
        对输入向量应用 softmax 函数，输出每个类的概率。
    """
    exp_vec = np.exp(vec - np.max(vec))  # 计算每个元素的指数，并防止溢出
    return exp_vec / exp_vec.sum(axis=0)  # 归一化，使输出和为 1

# 交叉熵损失函数
def cross_entropy(pred, target):
    """
    输入:
        pred: 模型的预测输出 (numpy array)
        target: 真实标签 (numpy array)
    输出:
        loss: 交叉熵损失 (float)
    功能:
        计算交叉熵损失。
    """
    num_samples = target.shape[0]
    return -np.sum(target * np.log(pred)) / num_samples

# 准确率计算
def calc_accuracy(predictions, targets):
    """
    输入:
        predictions: 模型预测的类别概率 (numpy array)
        targets: 真实的类别标签 (numpy array)
    输出:
        accuracy: 模型的准确率 (float)
    功能:
        根据预测的类别和真实的类别计算模型的准确率。
    """
    pred_labels = np.argmax(predictions, axis=1)  # 获取预测的类别标签
    true_labels = np.argmax(targets, axis=1)  # 获取真实的类别标签
    return np.mean(pred_labels == true_labels)  # 计算预测正确的比例

# 初始化卷积核权重和全连接层的权重、偏置
kernel_weights = np.random.rand(3, 3)  # 随机初始化卷积核
fc_weights = np.random.rand(169, 10)  # 随机初始化全连接层的权重
fc_bias = np.random.rand(10)  # 随机初始化全连接层的偏置

# 加载 MNIST 数据集并进行预处理
data, labels = fetch_openml('mnist_784', version=1, parser='auto', return_X_y=True)
data = data.to_numpy() / 255.0  # 将数据归一化为 0-1 范围
labels = np.array(labels)

# 将标签转换为 one-hot 编码
one_hot_labels = np.zeros((labels.shape[0], 10))
for idx in range(labels.shape[0]):
    one_hot_labels[idx, int(labels[idx])] = 1

# 将数据集分为训练集和测试集
train_data, test_data, train_labels, test_labels = train_test_split(data, one_hot_labels, test_size=0.2, random_state=42)
train_data = train_data.reshape((-1, 28, 28))  # 将数据调整为 28x28 图像
test_data = test_data.reshape((-1, 28, 28))

# CNN 前向传播过程
def cnn_forward_pass(input_image):
    """
    输入:
        input_image: 2D 输入图像 (numpy array)
    输出:
        final_out: 最终输出的类别概率 (numpy array)
        flattened: 池化后展平的输出 (numpy array)
    功能:
        执行卷积、激活、池化和全连接层，计算最终输出的类别概率。
    """
    conv_out = convolution(input_image, kernel_weights, stride=1, padding=0)  # 卷积操作
    activated = relu(conv_out)  # ReLU 激活
    pooled = max_pool(activated, pool_size=(2, 2), step=2)  # 最大池化
    flattened = pooled.flatten()  # 展平池化结果
    dense_out = fully_connected(flattened, fc_weights, fc_bias)  # 全连接层
    final_out = softmax(dense_out)  # 应用 softmax 输出类别概率
    return final_out, flattened

# 训练过程
def train_network(train_imgs, train_lbls, epochs, learning_rate):
    """
    输入:
        train_imgs: 训练图像数据 (numpy array)
        train_lbls: 训练标签 (numpy array)
        epochs: 训练轮次 (int)
        learning_rate: 学习率 (float)
    输出:
        loss_history: 每轮训练的损失 (list)
        accuracy_history: 每轮训练的准确率 (list)
    功能:
        执行网络的训练过程，更新权重，记录损失和准确率。
    """
    global fc_weights, fc_bias  # 使用全局变量来更新全连接层权重和偏置
    subset_imgs = train_imgs[:10000]  # 取训练集的子集
    subset_lbls = train_lbls[:10000]  # 取标签的子集
    loss_history = []  # 记录损失的历史
    accuracy_history = []  # 记录准确率的历史
    # 迭代训练
    for ep in range(epochs):
        total_loss = 0
        for idx in range(len(subset_imgs)):
            # 前向传播
            pred, flat_output = cnn_forward_pass(subset_imgs[idx])
            # 计算损失
            loss = cross_entropy(pred, subset_lbls[idx])
            total_loss += loss
            # 计算梯度并更新权重
            grad_error = pred - subset_lbls[idx]  # 损失的梯度
            grad_weights = np.outer(flat_output, grad_error)  # 计算权重梯度
            grad_bias = grad_error  # 偏置梯度
            # 梯度下降更新权重和偏置
            fc_weights -= learning_rate * grad_weights
            fc_bias -= learning_rate * grad_bias
        avg_loss = total_loss / len(subset_imgs)  # 计算平均损失
        # 计算训练集准确率
        train_acc = calc_accuracy(np.array([cnn_forward_pass(img)[0] for img in subset_imgs]), subset_lbls)
        loss_history.append(avg_loss)
        accuracy_history.append(train_acc)
        print(f"Epoch {ep + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {train_acc:.4f}")
    return loss_history, accuracy_history

# 模型评估
def evaluate_model(test_imgs, test_lbls):
    """
    输入:
        test_imgs: 测试图像数据 (numpy array)
        test_lbls: 测试标签 (numpy array)
    功能:
        评估模型在测试集上的准确率。
    """
    test_subset = test_imgs[:2000]  # 取测试集的子集
    lbl_subset = test_lbls[:2000]  # 取测试集标签的子集
    eval_acc = calc_accuracy(np.array([cnn_forward_pass(img)[0] for img in test_subset]), lbl_subset)
    print(f"Test Accuracy: {eval_acc:.4f}")

# 可视化预测结果
def plot_predictions(test_imgs, test_lbls):
    """
    输入:
        test_imgs: 测试图像数据 (numpy array)
        test_lbls: 测试标签 (numpy array)
    功能:
        可视化模型的预测结果，并标注预测正确或错误的样例。
    """
    imgs_subset = test_imgs[:36]  # 取前 36 张测试图像
    lbls_subset = test_lbls[:36]  # 取前 36 个标签
    preds = np.array([cnn_forward_pass(img)[0] for img in imgs_subset])  # 预测输出
    pred_classes = np.argmax(preds, axis=1)  # 获取预测类别
    true_classes = np.argmax(lbls_subset, axis=1)  # 获取真实类别
    fig, axes = plt.subplots(6, 6, figsize=(20, 20))  # 创建 6x6 的子图网格
    axes = axes.ravel()  # 将多维数组展平
    for i in range(36):
        axes[i].imshow(imgs_subset[i].reshape(28, 28), cmap='gray')  # 显示图像
        title_color = 'black' if pred_classes[i] == true_classes[i] else 'red'  # 如果预测正确，标题为黑色，否则为红色
        axes[i].set_title(f"True: {true_classes[i]}, Pred: {pred_classes[i]}", fontsize=8, color=title_color)
        axes[i].axis('off')  # 隐藏坐标轴
    plt.subplots_adjust(wspace=1, hspace=1)  # 调整子图间距

# 绘制训练指标 (损失和准确率)
def plot_training_metrics(losses, accuracies):
    """
    输入:
        losses: 每轮训练的损失历史 (list)
        accuracies: 每轮训练的准确率历史 (list)
    功能:
        绘制训练过程中的损失和准确率曲线。
    """
    epoch_range = range(1, len(losses) + 1)  # 获取训练轮次
    fig, ax1 = plt.subplots()  # 创建绘图区域
    ax1.set_xlabel('Epochs')  # 设置 X 轴标签
    ax1.set_ylabel('Loss', color='red')  # 设置 Y 轴 (损失) 标签和颜色
    ax1.plot(epoch_range, losses, color='red')  # 绘制损失曲线
    ax1.tick_params(axis='y', labelcolor='red')  # 设置 Y 轴颜色
    ax1.fill_between(epoch_range, losses, color='red', alpha=0.5)  # 填充损失曲线下方区域
    ax2 = ax1.twinx()  # 创建双 Y 轴
    ax2.set_ylabel('Accuracy', color='blue')  # 设置 Y 轴 (准确率) 标签和颜色
    ax2.plot(epoch_range, accuracies, color='blue')  # 绘制准确率曲线
    ax2.tick_params(axis='y', labelcolor='blue')  # 设置 Y 轴颜色
    ax2.fill_between(epoch_range, accuracies, color='green', alpha=0.5)  # 填充准确率曲线下方区域
    fig.tight_layout()  # 自动调整布局
    plt.title('Training Loss and Accuracy')  # 设置图表标题
    plt.show()  # 显示图表

if __name__ == "__main__":
    # 训练网络并可视化训练结果
    losses, accuracies = train_network(train_data, train_labels, epochs=30, learning_rate=0.001)
    plot_predictions(test_data, test_labels)
    plt.show()
    plot_training_metrics(losses, accuracies)