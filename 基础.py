import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

"""
与numpy 区别
data = [-1, -2, 1, 2]
tensor = torch.FloatTensor(data)

print(
    'numpy', np.abs(data),
    '\ntorch', torch.abs(tensor)
)
"""

"""
Variable


tensor = torch.FloatTensor([[1, 2], [3, 4]])
variable = Variable(tensor, requires_grad=True)  # require_gra反向传播时True会计算当前variable的gradient

t_out = torch.mean(tensor * tensor)   # x^2
v_out = torch.mean(variable * variable)

print(t_out)
print(v_out)

v_out.backward()  # variable 反向传递
# v_out = 1/4 * sum(var * var)
# d(v_out) / d(var) = 1/4 * 2 * variable = variable / 2
print(variable.grad) # 注意此时v_out backward后 variable的gradient也会变

print(variable)

print(variable.data)

print(variable.data.numpy()) #若要把variable转换成numpy则需要用.data来转换
"""

"""
Activation Function

# fake data
x = torch.linspace(-5, 5, 200)  # x data (tensor), shape=(100,1) 在-5到5区间中取200个点
x = Variable(x)
x_np = x.data.numpy()   # 换成 numpy array, 出图时用

# 几种常用的 激励函数
y_relu = F.relu(x).data.numpy()
y_sigmoid = F.sigmoid(x).data.numpy()
y_tanh = F.tanh(x).data.numpy()
y_softplus = F.softplus(x).data.numpy()
# y_softmax = F.softmax(x)  softmax 比较特殊, 不能直接显示, 不过他是关于概率的, 用于分类

# 图形化
plt.figure(1, figsize=(8, 6))
plt.subplot(221)
plt.plot(x_np, y_relu, c='red', label='relu')
plt.ylim((-1, 5))
plt.legend(loc='best')

plt.subplot(222)
plt.plot(x_np, y_sigmoid, c='red', label='sigmoid')
plt.ylim((-0.2, 1.2))
plt.legend(loc='best')

plt.subplot(223)
plt.plot(x_np, y_tanh, c='red', label='tanh')
plt.ylim((-1.2, 1.2))
plt.legend(loc='best')

plt.subplot(224)
plt.plot(x_np, y_softplus, c='red', label='softplus')
plt.ylim((-0.2, 6))
plt.legend(loc='best')

plt.show()
"""

"""
Regression

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1) unsqueeze把一维数据变成二维数据
y = x.pow(2) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)

# 画图
# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()

class Net(torch.nn.Module):  # 继承 torch 的 Module
    def __init__(self, n_feature, n_hidden, n_output):  # 神经元个数
        super(Net, self).__init__()     # 继承 __init__ 功能
        # 定义每层用什么样的形式
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # 隐藏层线性输出
        self.predict = torch.nn.Linear(n_hidden, n_output)   # 输出层线性输出

    def forward(self, x):   # 这同时也是 Module 中的 forward 功能
        # 正向传播输入值, 神经网络分析出输出值
        x = F.relu(self.hidden(x))      # 激励函数(隐藏层的线性值)
        x = self.predict(x)             # 输出值
        return x

net = Net(n_feature=1, n_hidden=10, n_output=1)  # 一个x, 10个神经元, 1个y输出

print(net)  # net 的结构
'''
Net (
  (hidden): Linear (1 -> 10)
  (predict): Linear (10 -> 1)
)
'''

# optimizer 是训练的工具
optimizer = torch.optim.SGD(net.parameters(), lr=0.2)  # 传入 net 的所有参数, lr = learning rate学习率,数值越高越快
loss_func = torch.nn.MSELoss()      # 预测值和真实值的误差计算公式 (均方差) MSELoss = mean square function 回归使用方差即可

# for t in range(100):  # 训练100步
#     prediction = net(x)     # 喂给 net 训练数据 x, 输出预测值
#
#     loss = loss_func(prediction, y)     # 计算两者的误差 prediction在前, y在后
#
#     optimizer.zero_grad()   # 清空上一步的残余更新参数值, 将梯度重置为0
#     loss.backward()         # 误差反向传播, 计算参数更新值
#     optimizer.step()        # 将参数更新值施加到 net 的 parameters 上

# 可视化训练
plt.ion()   # 画图
plt.show()

for t in range(200):

    prediction = net(x)     # 喂给 net 训练数据 x, 输出预测值

    loss = loss_func(prediction, y)     # 计算两者的误差 prediction在前, y在后

    optimizer.zero_grad()   # 清空上一步的残余更新参数值, 将梯度重置为0
    loss.backward()         # 误差反向传播, 计算参数更新值
    optimizer.step()        # 将参数更新值施加到 net 的 parameters 上
    # 接着上面来
    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)
plt.ioff()  # 画图
plt.show()
"""

"""
分类
"""

# 假数据
n_data = torch.ones(100, 2)         # 数据的基本形态
x0 = torch.normal(2*n_data, 1)      # 类型0 x data (tensor), shape=(100, 2)
y0 = torch.zeros(100)               # 类型0 y data (tensor), shape=(100, )
x1 = torch.normal(-2*n_data, 1)     # 类型1 x data (tensor), shape=(100, 1)
y1 = torch.ones(100)                # 类型1 y data (tensor), shape=(100, )

# 注意 x, y 数据的数据形式是一定要像下面一样 (torch.cat 是在合并数据)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # FloatTensor = 32-bit floating
y = torch.cat((y0, y1), ).type(torch.LongTensor)    # LongTensor = 64-bit integer

# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
# plt.show()

# 画图
# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()

class Net(torch.nn.Module):     # 继承 torch 的 Module
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()     # 继承 __init__ 功能
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # 隐藏层线性输出
        self.out = torch.nn.Linear(n_hidden, n_output)       # 输出层线性输出

    def forward(self, x):
        # 正向传播输入值, 神经网络分析出输出值
        x = F.relu(self.hidden(x))      # 激励函数(隐藏层的线性值)
        x = self.out(x)                 # 输出值, 但是这个不是预测值, 预测值还需要再另外计算
        return x

net = Net(n_feature=2, n_hidden=10, n_output=2) # 2个特征，几个类别就几个 output

print(net)  # net 的结构
'''
Net (
  (hidden): Linear (2 -> 10)
  (out): Linear (10 -> 2)
)
'''

# optimizer 是训练的工具
optimizer = torch.optim.SGD(net.parameters(), lr=0.02)  # 传入 net 的所有参数, 学习率
# 算误差的时候, 注意真实值!不是! one-hot 形式的, 而是1D Tensor, (batch,)
# 但是预测值是2D tensor (batch, n_classes)
loss_func = torch.nn.CrossEntropyLoss()

# for t in range(100):
#     out = net(x)     # 喂给 net 训练数据 x, 输出分析值
#
#     loss = loss_func(out, y)     # 计算两者的误差
#
#     optimizer.zero_grad()   # 清空上一步的残余更新参数值
#     loss.backward()         # 误差反向传播, 计算参数更新值
#     optimizer.step()        # 将参数更新值施加到 net 的 parameters 上

plt.ion()   # 画图
plt.show()

for t in range(100):

    out = net(x)  # 喂给 net 训练数据 x, 输出分析值

    loss = loss_func(out, y)  # 计算两者的误差

    optimizer.zero_grad()  # 清空上一步的残余更新参数值
    loss.backward()
    optimizer.step()

    # 接着上面来
    if t % 2 == 0:
        plt.cla()
        # 过了一道 softmax 的激励函数后的最大概率才是预测值
        prediction = torch.max(F.softmax(out), 1)[1]
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y)/200.  # 预测中有多少和真实值一样
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

plt.ioff()  # 停止画图
plt.show()