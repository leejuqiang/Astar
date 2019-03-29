import gym
import torch
import math


class Net(torch.nn.Module):

    def __init__(self, inputN, layers, outputN):
        super(Net, self).__init__()
        self.rate = 0
        self.opt = None
        self.cacheP = None
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

    def cacheParameter(self):
        self.cacheP = []
        for t in list(self.parameters()):
            l = t.tolist()
            self.cacheP.append(l.copy())

    def applyCachedParameter(self):
        i = 0
        for h in self.hidden:
            h.weight.data = torch.tensor(
                self.cacheP[i]).type(torch.FloatTensor)
            h.bias.data = torch.tensor(
                self.cacheP[i + 1]).type(torch.FloatTensor)
            i += 2
        self.out.weight.data = torch.tensor(
            self.cacheP[i]).type(torch.FloatTensor)
        self.out.bias.data = torch.tensor(
            self.cacheP[i + 1]).type(torch.FloatTensor)

    def initParam(self, rate):
        self.rate = rate
        self.opt = torch.optim.SGD(self.parameters(), lr=self.rate)

    def train(self, x, y, times):
        lossFun = torch.nn.MSELoss()
        for i in range(times):
            out = self(x)
            loss = lossFun(out, y)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()


nn = Net(4, [10], 1)

isTrain = True


def getAct(ob):
    x = torch.tensor(ob).type(torch.FloatTensor)
    result = nn(x).tolist()
    act = 1 if result[0] > 0 else 0
    return result, act


def score(env):
    count = 0
    for ep in range(50):
        ob = env.reset()
        for i in range(10000):
            env.render()
            res, act = getAct(ob)
            ob, reward, done, info = env.step(act)
            if done:
                count += i
                break
    return count


def train(x, acts, outX, outY):
    lastAng = x[len(x) - 1][2]
    for i in range(len(x) - 1, -1, -1):
        if (x[i][2] < 0 and lastAng > 0) or (x[i][2] > 0 and lastAng < 0):
            break
        outX.append(x[i])
        y = -10 if acts[i] > 0 else 10
        outY.append(y)


def run(runTimes):
    maxScore = 0
    trainCount = 0
    env = gym.make('CartPole-v0')
    count = 0
    trainX = None
    trainActs = None
    inputX = []
    inputY = []
    rate = 0.01
    for ep in range(runTimes):
        ob = env.reset()
        if isTrain:
            trainActs = []
            trainX = []
        for i in range(10000):
            env.render()
            result, act = getAct(ob)
            if isTrain:
                trainX.append(ob.copy())
                trainActs.append(result[0])
            ob, reward, done, info = env.step(act)
            if done:
                count += i
                if isTrain:
                    trainCount += 1
                    if trainCount >= 9:
                        train(trainX, trainActs, inputX, inputY)
                        x = torch.tensor(inputX).type(torch.FloatTensor)
                        y = torch.tensor(inputY).type(torch.FloatTensor)
                        nn.initParam(rate)
                        nn.train(x, y, 200)
                        inputX = []
                        inputY = []
                        trainCount = 0
                        rate *= 0.5
                        # s = score(env)
                        # if s < maxScore:
                        #     nn.applyCachedParameter()
                        # else:
                        #     maxScore = s
                        #     nn.cacheParameter()
                break
    env.close()
    if not isTrain:
        count /= runTimes
        print(count)
    else:
        print("finish train")


def printWeight():
    for k, v in nn.state_dict().items():
        if "weight" in k:
            print(v)


def test():
    global isTrain
    isTrain = True
    run(200)
    isTrain = False
    run(100)


test()


# 20
