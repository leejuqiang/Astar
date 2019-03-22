import gym
import torch
import numpy as np
import torch.nn.functional as tf
from torch.autograd import Variable


class Stick:
    def __init__(self):
        self.env = gym.make('CartPole-v0')

    def getAct(self, isTrain):
        return self.env.action_space.sample()

    def stepData(self, ob, reward, isTrain):
        pass

    def run(self, isTrain):
        count = 0
        for ep in range(100):
            ob = self.env.reset()
            for i in range(1000):
                self.env.render()
                act = self.getAct()
                ob, reward, done, info = self.env.step(act)
                self.stepData(ob, reward)
                if done:
                    count += i
                    break
        self.env.close()
        print(count / 100)


class StickHardCore(Stick):

    lastAng = 0

    def getAct(self, isTrain):
        if self.lastAng <= 0:
            return 0
        else:
            return 1

    def stepData(self, ob, reward, isTrain):
        self.lastAng = ob[2]


class Net(torch.nn.Module):
    rate = 0
    opt = None

    def __init__(self, inputN, layers, outputN):
        super(Net, self).__init__()
        self.hidden = torch.nn.ModuleList()
        last = inputN
        for l in layers:
            self.hidden.append(torch.nn.Linear(last, l))
            last = l
        self.out = torch.nn.Linear(last, outputN)

    def forward(self, x):
        for l in self.hidden:
            x = torch.sigmoid(l(x))
        x = torch.sigmoid(self.out(x))
        return x

    def initParam(self, rate):
        self.rate = rate
        self.opt = torch.optim.SGD(self.parameters(), lr=self.rate)

    def train(self, x, y):
        lossFun = torch.nn.CrossEntropyLoss()
        for i in range(200):
            out = self(x)
            o = torch.cat((out, 1 - out), 1)
            loss = lossFun(o, y)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()


# 分别生成2组各100个数据点，增加正态噪声，后标记以y0=0 y1=1两类标签，最后cat连接到一起
n_data = torch.ones(100, 2)
# torch.normal(means, std=1.0, out=None)
x0 = torch.normal(2*n_data, 1)  # 以tensor的形式给出输出tensor各元素的均值，共享标准差
y0 = torch.zeros(100)
x1 = torch.normal(-2*n_data, 1)
y1 = torch.ones(100)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # 组装（连接）
y = torch.cat((y0, y1), 0).type(torch.LongTensor)
# x, y = Variable(x), Variable(y)

net = Net(2, [10], 1)
net.initParam(0.05)
net.train(x, y)

pre = net(x)
count = 0
for i in range(200):
    p = pre[i]
    if y[i] == 0 and p.data[0] > 0.5:
        count += 1
    if y[i] == 1 and p.data[0] <= 0.5:
        count += 1
print(count)
# test = torch.tensor([-1, -1]).type(torch.FloatTensor)
# print(net(test))


class StickNN(Stick):

    def __init__(self):
        return super().__init__(4, 20, 1)

    def getAct(self, isTrain):
        return 0

    def stepData(self, ob, reward, isTrain):
        pass

# print("random:")
# stickRandom = Stick()
# for i in range(3):
#     stickRandom.run(False)
# random:
# 22.39
# 22.22
# 22.49

# print("hard core:")
# stickHard = StickHardCore()
# for i in range(3):
#     stickHard.run(False)
# hard core:
# 43.08
# 41.34
# 41.79


# stick100 = StickNN()
# for i in range(3):
#     stick100.run(False)
