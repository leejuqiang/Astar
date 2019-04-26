import gym
import torch

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

f = open("log.txt", "w")
def log(s):
    global f
    s += "\n"
    f.write(s)
    print(s)

env = gym.make('Pong-ram-v0')
nn = Net(128, [10, 10], 2)
nn.initParam(0.001)

def changeColor(rgb):
    return rgb[0] * 0.299 + rgb[1] * 0.587 + rgb[2] * 0.114

def obToState(ob):
    return ob;
    # x = []
    # for i in range(210):
    #     for j in range(160):
    #         x.append(changeColor(ob[i][j]))
    # return x

def getAction(res):
    return 2 if res[0] > res[1] else 3

def getOutput(state):
    x = torch.tensor(state).type(torch.FloatTensor)
    res = nn(x).tolist()
    return res

def train(runTimes, epoch):
    for j in range(epoch):
        total = 0
        x = []
        y = []
        while total < runTimes:
            acts = []
            outputs = []
            obs = []
            ob = env.reset()
            for i in range(10000):
                env.render(mode="rgb_array")
                state = obToState(ob)
                obs.append(state)
                out = getOutput(state)
                act = getAction(out)
                acts.append(act)
                outputs.append(out)
                ob, reward, done, info = env.step(act)
                if reward != 0:
                    total += 1
                    r = -10
                    if reward > 0:
                        r = -r
                    for i in range(len(obs)):
                        if acts[i] == 2:
                            outputs[i][0] += r
                            outputs[i][1] -= r
                        else:
                            outputs[i][0] -= r
                            outputs[i][1] += r
                        # if reward < 0:
                        #     outputs[i][0], outputs[i][1] = outputs[i][1], outputs[i][0]
                        y.append(outputs[i])
                        x.append(obs[i])
                        r *= 0.95
                if done:
                    break
        x = torch.tensor(x).type(torch.FloatTensor)
        y = torch.tensor(y).type(torch.FloatTensor)
        nn.train(x, y, 100)
        log("finish train " + str(j))

def run(runTimes):
    count = 0
    total = 0
    while total < runTimes:
        ob = env.reset()
        for i in range(10000):
            env.render(mode="rgb_array")
            state = obToState(ob)
            out = getOutput(state)
            act = getAction(out)
            ob, reward, done, info = env.step(act)
            if reward != 0:
                total += 1
                if reward > 0:
                    count += 1
            if done:
                break
    env.close()
    log("win " + str(count) + " total " + str(runTimes))

train(100, 200)

nn.cacheParameter()
for p in nn.cacheP:
    log(str(p))

for i in range(3):
    run(200)


# win 6 total 200
# win 11 total 200
# win 4 total 200
f.close()