import gym
import torch

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

nn = Net(4, [10], 1)
nn.initParam(0.01)

isTrain = True

def train(x, acts):
    if not isTrain:
        return
    y = [0 for i in range(len(x))]
    lastAng = x[len(x) - 1][2]
    invert = True
    f = 1
    for i in range(len(x) - 1, -1, -1):
        if not invert and ((x[i][2] < 0 and lastAng > 0) or (x[i][2] > 0 and lastAng < 0)):
            invert = False
        if not invert:
            y[i] = -1 if acts[i] == 0 else 1
        else:
            y[i] = 10 if acts[i] == 0 else -10
            # y[i] *= f
            # f *= 0.95
    x = torch.tensor(x).type(torch.FloatTensor)
    y = torch.tensor(y).type(torch.FloatTensor)
    nn.train(x, y, 200)

def run(runTimes):
    env = gym.make('CartPole-v0')
    count = 0
    trainX = None
    trainActs = None
    for ep in range(runTimes):
        ob = env.reset()
        if isTrain:
            trainActs = []
            trainX = []
        for i in range(10000):
            env.render()
            x = torch.tensor(ob).type(torch.FloatTensor)
            act = nn(x).tolist()
            act = 1 if act[0] > 0 else 0
            if isTrain:
                trainX.append(ob.copy())
                trainActs.append(act)
            ob, reward, done, info = env.step(act)
            if done:
                count += i
                train(trainX, trainActs)
                break
    env.close()
    if not isTrain:
        print(count / runTimes)
    else:
        print("finish train")

def printWeight():
    for k, v in nn.state_dict().items():
        if "weight" in k:
            print(v)

run(200)

# 100
# 42.65
# 41.96
# 44.43

# 200
# 115.24
# 113.19
# 110.63
isTrain = False
for i in range(3):
    run(100)