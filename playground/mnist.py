import numpy as np
import torch
from mlxtend.data import loadlocal_mnist
from torch import nn

x_train, y_train_label = loadlocal_mnist(
    images_path='./dataset/mnist/train-images-idx3-ubyte',
    labels_path='./dataset/mnist/train-labels-idx1-ubyte'
)
# print(x_train.shape[0])
# print(y_train_label[:10])
x = torch.tensor(y_train_label[:5], dtype=torch.int64)
y = torch.nn.functional.one_hot(x, 10)
# print(y)

# 简单来说，MNIST数据集的标签实际上就是一个表示60 000幅图片的60 000×10大小的矩阵张量[60000,10]。
# 前面的行数指的是数据集中的图片为60 000幅，后面的10是指10个列向量。
# tensor([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
# 因此为了实现对输入图像进行数字分类这个想法，必须设计一个合适的判别模型。
# 而从上面对图像的分析来看，最直观的想法就将图形作为一个整体结构直接输入到模型中进行判断。
#

# 实现一个多层感知机
class NeuralNetWork(nn.Module):
    def __init__(self):
        super(NeuralNetWork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 312),
            nn.ReLU(),
            nn.Linear(312, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, input):
        x = self.flatten(input)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetWork()
model = model.to('cpu')
model = torch.compile(model)
loss_fn = nn.CrossEntropyLoss()
# 采用 Adam
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

# 一些配置变量
batch_size = 320  # 设定每次的训练批数
epochs = 1024  # 设定训练次数
train_num = len(x_train) // batch_size
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for epoch in range(20):
    train_loss = 0
    for i in range(batch_size):
        start = i * batch_size
        end = (i + 1) * batch_size
        train_batch = torch.tensor(x_train[start:end]).to(device)
        label_batch = torch.tensor(y_train_label[start:end]).to(device)
        pred = model(train_batch)
        loss = loss_fn(pred, label_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_loss /= train_num
    accuracy = (pred.argmax(dim=1) == label_batch).type(torch.float32).sum().item() / batch_size
    print('train loss: {}, train accuracy: {}'.format(train_loss, accuracy))
