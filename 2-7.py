# coding=utf-8
import random
import numpy as np


class Network(object):
    def __init__(self, sizes):
        # 网络层数
        self.num_layers = len(sizes)
        # 网络每层神经元个数
        self.sizes = sizes
        # 初始化每层的偏置
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # 初始化每层的权重
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    # 梯度下降
    def GD(self, training_data, epochs):
        # 开始训练 循环每一个epochs
        for j in xrange(epochs):
            # 洗牌 打乱训练数据
            random.shuffle(training_data)

            # 训练每一个数据
            for x, y in training_data:
                self.update(x, y)

            print "Epoch {0} complete".format(j)

    # 前向传播
    def update(self, x, y):
        activation = x

        # 保存每一层的激励值a=sigmoid(z)
        activations = [x]

        # 保存每一层的z=wx+b
        zs = []
        # 前向传播
        for b, w in zip(self.biases, self.weights):
            # 计算每层的z
            z = np.dot(w, activation) + b

            # 保存每层的z
            zs.append(z)

            # 计算每层的a
            activation = sigmoid(z)

            # 保存每一层的a
            activations.append(activation)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


