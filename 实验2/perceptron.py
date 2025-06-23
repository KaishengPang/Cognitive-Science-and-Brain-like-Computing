import numpy as np

# 定义感知器类
class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=1000):
        # 初始化权重，input_size+1 表示输入向量的长度+1（包括偏置项）
        self.W = np.zeros(input_size + 1)  # 初始权重为0
        self.learning_rate = learning_rate  # 学习率
        self.epochs = epochs  # 训练迭代次数

    # 定义激活函数，使用阶跃函数，将大于或等于0的输出定义为1，否则为0
    def activation_fn(self, x):
        return 1 if x >= 0 else 0

    # 定义感知器的预测函数
    def predict(self, x):
        # 使用当前的权重进行预测，计算线性组合 W.T * X（包括偏置）
        z = self.W.T.dot(np.insert(x, 0, 1))  # np.insert用于在x向量前插入1以表示偏置项
        a = self.activation_fn(z)  # 将结果通过激活函数
        return a

    # 训练感知器
    def fit(self, X, d):
        # 进行多轮迭代训练
        for _ in range(self.epochs):
            # 对每个样本依次更新权重
            for i in range(len(X)):
                x_i = np.insert(X[i], 0, 1)  # 在输入向量前插入1以表示偏置项
                y_hat = self.predict(X[i])  # 预测当前输入的输出
                error = d[i] - y_hat  # 计算预测值与真实值的误差
                self.W = self.W + self.learning_rate * error * x_i  # 更新权重

# 真值表构建（与运算），输入是[0,0], [0,1], [1,0], [1,1]，目标输出是[0,0,0,1]
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])

# 真值表构建（或运算），输入是[0,0], [0,1], [1,0], [1,1]，目标输出是[0,1,1,1]
X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_or = np.array([0, 1, 1, 1])

# 真值表构建（非运算），输入是[0], [1]，目标输出是[1,0]
X_not = np.array([[0], [1]])
y_not = np.array([1, 0])

# 创建感知器对象，用于分别训练与、或和非运算
perceptron_and = Perceptron(input_size=2)  # 输入为2个
perceptron_or = Perceptron(input_size=2)  # 输入为2个
perceptron_not = Perceptron(input_size=1)  # 输入为1个

# 训练感知器，分别训练与运算、或运算、非运算的感知器
perceptron_and.fit(X_and, y_and)
perceptron_or.fit(X_or, y_or)
perceptron_not.fit(X_not, y_not)

# 测试感知器的预测结果
print("与运算perceptron预测:")
for x in X_and:
    print(f"输入: {x}, 输出: {perceptron_and.predict(x)}")  # 输出与运算的预测结果

print("\n或运算perceptron预测:")
for x in X_or:
    print(f"输入: {x}, 输出: {perceptron_or.predict(x)}")  # 输出或运算的预测结果

print("\n非运算perceptron预测:")
for x in X_not:
    print(f"输入: {x}, 输出: {perceptron_not.predict(x)}")  # 输出非运算的预测结果