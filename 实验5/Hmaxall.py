import numpy as np  
import cv2    
import torch    
from torchvision import datasets, transforms  
from sklearn.svm import SVC  # 导入 SVM 分类器  
from sklearn.metrics import accuracy_score  # 导入准确率评估函数  
import matplotlib.pyplot as plt   
from tqdm import tqdm  # 导入 tqdm 库，用于显示循环的进度条  
from sklearn.decomposition import PCA  # 导入 PCA 用于降维  
  
# Gabor 滤波器生成函数    
def gabor_fn(sigma, theta, Lambda, psi, gamma):    
    # 计算滤波器的标准差  
    sigma_x = sigma  # x 方向的标准差  
    sigma_y = sigma / gamma  # y 方向的标准差，由 gamma 控制  
    
    # 构建滤波器窗口大小  
    nstds = 3  # 窗口大小为标准差的倍数  
    xmax = np.ceil(max(1, abs(nstds * sigma_x * np.cos(theta))))  # 计算窗口的最大 x 值  
    ymax = np.ceil(max(1, abs(nstds * sigma_y * np.sin(theta))))  # 计算窗口的最大 y 值  
    xmin = -xmax  # 窗口的最小 x 值  
    ymin = -ymax  # 窗口的最小 y 值  
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))  # 生成网格坐标  
    
    # 生成 Gabor 滤波器核函数  
    x_theta = x * np.cos(theta) + y * np.sin(theta)  # 计算旋转后的 x 坐标  
    y_theta = -x * np.sin(theta) + y * np.cos(theta)  # 计算旋转后的 y 坐标  
    
    # 计算 Gabor 滤波器的值  
    gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) *  np.cos(2 * np.pi / Lambda * x_theta + psi)  # 计算 Gabor 核的复数部分  
    return gb  # 返回 Gabor 滤波器  
    
# 生成 Gabor 滤波器组    
def create_gabor_filters():    
    filters = []  # 初始化滤波器组的列表  
    # 从 4 个方向生成滤波器  
    for theta in np.arange(0, np.pi, np.pi / 4):  # theta 从 0 到 π，共有 4 个方向  
        # 在 16 个不同的尺度生成滤波器  
        for scale in np.linspace(3, 15, 16):  # 生成 16 个不同尺度的滤波器  
            kern = gabor_fn(sigma=1, theta=theta, Lambda=scale, psi=0, gamma=0.5)  # 生成 Gabor 滤波器  
            filters.append(kern)  # 将生成的滤波器添加到列表中  
    return filters  # 返回滤波器组  
    
# 应用 Gabor 滤波器到图像    
def apply_gabor_filters(image, filters):    
    responses = []  # 初始化滤波响应的列表  
    for kern in filters:  # 遍历每个滤波器  
        # 使用 OpenCV 对图像进行滤波处理  
        response = cv2.filter2D(image, cv2.CV_32F, kern)  # 应用 2D 滤波器  
        responses.append(response)  # 将滤波结果添加到响应列表中  
    return responses  # 返回所有滤波器的响应  
    
# C1 层：池化并合并尺寸    
def max_pooling_and_combine(responses, pool_size=(8, 8)):    
    pooled_responses = []  # 初始化池化响应的列表  
    # 进行池化（缩小特征图尺寸）  
    for response in responses:  # 遍历每个滤波器的响应  
        pooled = cv2.resize(response, (response.shape[1] // pool_size[0], response.shape[0] // pool_size[1]))  # 缩小响应图  
        pooled_responses.append(pooled)  # 添加池化后的响应  
    
    # 合并相邻的尺寸响应取最大值  
    combined_responses = []  # 初始化合并响应的列表  
    for i in range(0, len(pooled_responses), 2):  # 每两个池化响应进行合并  
        combined = np.maximum(pooled_responses[i], pooled_responses[i + 1])  # 取最大值合并响应  
        combined_responses.append(combined)  # 添加合并后的响应  
    
    return combined_responses  # 返回合并后的响应  
    
# S2 层：新一轮滤波器应用    
def s2_layer(pooled_responses, num_directions=8):    
    s2_responses = []  # 初始化 S2 层响应的列表  
    num_responses = len(pooled_responses)  # 获取池化响应的数量  
        
    # 对每个 pooled_responses 进行处理  
    for i in range(len(pooled_responses)):  # 遍历每个池化响应  
        # 随机选取 num_directions 个方向  
        selected_indices = np.random.choice(num_responses, num_directions, replace=False)  # 随机选择方向  
        selected_responses = [pooled_responses[idx] for idx in selected_indices]  # 获取选中的响应  
        # 对选取的特征图取均值  
        avg_response = np.mean(selected_responses, axis=0)  # 计算均值响应  
        s2_responses.append(avg_response)  # 添加均值响应  
    
    return s2_responses  # 返回 S2 层的响应  
    
# 将所有特征平层为向量    
def flatten_responses(pooled_responses):    
    flattened = []  # 初始化平展特征的列表  
    for response in pooled_responses:  # 遍历每个池化响应  
        flattened.append(response.flatten())  # 将响应平展为一维数组并添加到列表中  
    return np.concatenate(flattened)  # 将所有平展后的特征连接成一个向量  
    
# 下载 MNIST 数据集    
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])  # 定义数据转换，包括归一化  
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)  # 下载训练集  
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)  # 下载测试集  
    
# 取样本用于实验    
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=600, shuffle=True)  # 加载训练数据，批大小为 600  
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)  # 加载测试数据，批大小为 100  
    
# 生成 Gabor 滤波器组    
gabor_filters = create_gabor_filters()  # 调用函数生成 Gabor 滤波器组  
    
# 提取训练集的 HMAX 特征    
train_features = []  # 初始化训练集特征列表  
train_labels = []  # 初始化训练集标签列表  
print("开始提取训练集的 HMAX 特征...")  # 打印开始提取特征的信息  
for images, labels in tqdm(train_loader, desc="Processing Training Data"):  # 遍历训练数据加载器  
    for i in range(images.shape[0]):  # 遍历每个批次中的图像  
        img = images[i].numpy().squeeze()  # 转换为二维数组（去掉颜色维度）  
        responses = apply_gabor_filters(img, gabor_filters)  # S1 层：应用 Gabor 滤波器  
        c1_responses = max_pooling_and_combine(responses)  # C1 层：池化并合并  
        s2_responses = s2_layer(c1_responses)  # S2 层：随机选取方向并计算均值  
        feature_vector = flatten_responses(s2_responses)  # 平展特征向量  
        train_features.append(feature_vector)  # 添加特征向量到训练特征列表  
        train_labels.append(labels[i].item())  # 添加标签到训练标签列表  
    
train_features = np.array(train_features)  # 将训练特征转换为 NumPy 数组  
train_labels = np.array(train_labels)  # 将训练标签转换为 NumPy 数组  
print("训练集特征提取完成！")  # 打印训练特征提取完成的信息  
    
# 提取测试集的 HMAX 特征    
test_features = []  # 初始化测试集特征列表  
test_labels = []  # 初始化测试集标签列表  
print("开始提取测试集的 HMAX 特征...")  # 打印开始提取测试集特征的信息  
for images, labels in tqdm(test_loader, desc="Processing Test Data"):  # 遍历测试数据加载器  
    for i in range(images.shape[0]):  # 遍历每个批次中的图像  
        img = images[i].numpy().squeeze()  # 转换为二维数组  
        responses = apply_gabor_filters(img, gabor_filters)  # S1 层：应用 Gabor 滤波器  
        c1_responses = max_pooling_and_combine(responses)  # C1 层：池化并合并  
        s2_responses = s2_layer(c1_responses)  # S2 层：随机选取方向并计算均值  
        feature_vector = flatten_responses(s2_responses)  # 平展特征向量  
        test_features.append(feature_vector)  # 添加特征向量到测试特征列表  
        test_labels.append(labels[i].item())  # 添加标签到测试标签列表  
    
# 将特征转换为数组    
test_features = np.array(test_features)  # 将测试特征转换为 NumPy 数组  
test_labels = np.array(test_labels)  # 将测试标签转换为 NumPy 数组  
print("测试集特征提取完成！")  # 打印测试集特征提取完成的信息  
    
# 使用 PCA 降维    
pca = PCA(n_components=20)  # 选择保留 20 个主成分  
train_features_pca = pca.fit_transform(train_features)  # 对训练特征进行 PCA 降维  
test_features_pca = pca.transform(test_features)  # 对测试特征进行 PCA 降维  
    
# 使用 SVM 进行分类    
svm = SVC(kernel='linear')  # 初始化线性核的 SVM 分类器  
print("SVM training")  # 打印 SVM 训练的信息  
svm.fit(train_features_pca, train_labels)  # 训练 SVM 分类器  
    
# 在测试集上进行预测    
print("SVM predicting")  # 打印 SVM 预测的信息  
predictions = svm.predict(test_features_pca)  # 使用 SVM 对测试集进行预测  
accuracy = accuracy_score(test_labels, predictions)  # 计算测试集上的准确率  
    
print(f'测试集上的准确率: {accuracy * 100:.2f}%')  # 打印测试集上的准确率  
    
# 随机选择一些测试集样本进行可视化    
num_samples = 10  # 可视化 10 个样本  
sample_indices = np.random.choice(len(test_features), num_samples, replace=False)  # 随机选择 10 个样本  
    
sample_images = test_dataset.data[sample_indices]  # 获取测试集中的样本图像  
sample_labels = test_labels[sample_indices]  # 获取对应的真实标签  
predicted_labels = svm.predict(test_features_pca[sample_indices])  # 获取对应的预测标签  
    
# 可视化真实标签与预测标签的对比    
fig, axs = plt.subplots(2, 5, figsize=(12, 6))  # 创建 2 行 5 列的图像网格  
fig.suptitle('MNIST Classification Results (True vs Predicted)', fontsize=16)  # 设置图像的标题  
    
for i, idx in enumerate(sample_indices):  # 遍历每个选择的样本  
    ax = axs[i // 5, i % 5]  # 获取网格中的子图位置  
    ax.imshow(sample_images[i], cmap='gray')  # 显示图像，使用灰度颜色映射  
    ax.set_title(f"True: {sample_labels[i]}\nPred: {predicted_labels[i]}")  # 设置子图的标题，显示真实与预测标签  
    ax.axis('off')  # 关闭坐标轴  
    
plt.tight_layout(rect=[0, 0, 1, 0.95])  # 调整布局，防止重叠  
plt.show()  # 显示图像