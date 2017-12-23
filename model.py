from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # 定义conv1函数的是图像卷积函数：输入为图像（1个频道，即灰度图）,输出为 6张特征图, 卷积核为5x5正方形
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        # 定义conv2函数的是图像卷积函数：输入为6张特征图,输出为16张特征图, 卷积核为5x5正方形
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()#随机选择输入的信道，将其设为0
        # 定义fc1（fullconnect）全连接函数1为线性函数：y = Wx + b，并将16*4*4个节点连接到120个节点上
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        # 定义fc2（fullconnect）全连接函数2为线性函数：y = Wx + b，并将120个节点连接到84个节点上
        self.fc2 = nn.Linear(120, 84)
        # 定义fc3（fullconnect）全连接函数3为线性函数：y = Wx + b，并将84个节点连接到10个节点上
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))  # conv->max_pool->relu
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # conv->dropout->max_pool->relu
        x = x.view(-1, x.size()[1] * x.size()[2] * x.size()[3])
        # 输入x经过全连接1，再经过ReLU激活函数，然后更新x
        x = F.relu(self.fc1(x))
        # 输入x经过全连接2，再经过ReLU激活函数，然后更新x
        x = F.relu(self.fc2(x))
        # x = F.dropout(x, training=self.training)#dropout
        # 输入x经过全连接3，然后更新x
        x = self.fc3(x)
        return x  # F.log_softmax(x)  or x
