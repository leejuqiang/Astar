import gym
import torch
import numpy as np
import torch.nn.functional as tf
from torch.autograd import Variable


class Stick:
    runTimes = 100

    def __init__(self):
        self.env = gym.make('CartPole-v0')

    def getAct(self, isTrain):
        return self.env.action_space.sample()

    def stepData(self, ob, reward, done, isTrain):
        pass
    
    def initRun(self, isTrain):
        pass

    def run(self, isTrain):
        count = 0
        for ep in range(self.runTimes):
            ob = self.env.reset()
            self.initRun(isTrain)
            for i in range(1000):
                self.env.render()
                act = self.getAct(isTrain)
                ob, reward, done, info = self.env.step(act)
                self.stepData(ob, reward, done, isTrain)
                if done:
                    count += i
                    break
        self.env.close()
        if not isTrain:
            print(count / self.runTimes)


class StickHardCore(Stick):

    lastAng = 0

    def getAct(self, isTrain):
        if self.lastAng <= 0:
            return 0
        else:
            return 1

    def stepData(self, ob, reward, done, isTrain):
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
        x = self.out(x)
        return x

    def initParam(self, rate):
        self.rate = rate
        self.opt = torch.optim.SGD(self.parameters(), lr=self.rate)

    def train(self, x, y, times):
        lossFun = torch.nn.MSELoss()
        for i in range(times):
            out = self(x)
            # o = torch.cat((out, 1 - out), 1)
            loss = lossFun(out, y)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

class StickNN(Stick):

    x = None
    y = None
    acts = None
    lastOb = [0, 0, 0, 0]

    def __init__(self):
        Stick.__init__(self)
        self.nn = Net(4, [10], 1)
        self.nn.initParam(0.01)

    def getAct(self, isTrain):
        input = torch.tensor(self.lastOb).type(torch.FloatTensor)
        act = self.nn(input).tolist()
        act = 1 if act[0] > 0 else 0
        if isTrain:
            self.acts.append(act)
        return act
    
    def initRun(self, isTrain):
        self.lastOb = [0, 0, 0, 0]
        self.x = []
        self.acts = []
        self.x.append(self.lastOb)

    def stepData(self, ob, reward, done, isTrain):
        self.lastOb = ob
        if not isTrain:
            return
        if not done:
            self.x.append(ob)
        else:
            self.y = [0 for i in range(len(self.x))]
            lastAng = self.x[len(self.x) - 1][2]
            setOne = False
            factor = 1
            for i in range(len(self.x) - 1, -1, -1):
                if not setOne and ((self.x[i][2] < 0 and lastAng > 0) or (self.x[i][2] > 0 and lastAng < 0)):
                    setOne = True
                if setOne:
                    self.y[i] = -1 if self.acts[i] == 0 else 1
                else:
                    self.y[i] = -10 * \
                        factor if self.acts[i] == 1 else 10 * factor
                    factor *= 0.9
            x = torch.tensor(self.x).type(torch.FloatTensor)
            y = torch.tensor(self.y).type(torch.FloatTensor)
            self.nn.train(x, y, 200)

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


snn = StickNN()
# snn.run(True)
# print("nn run 100")
# for i in range(3):
#     snn.run(False)
# nn run 100
# 34.77
# 34.54
# 34.74

snn.runTimes = 200
snn.run(True)
print("nn run 200")
snn.runTimes = 100
for i in range(3):
    snn.run(False)
# nn run 200
# 58.42
# 60.83
# 62.32