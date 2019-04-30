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

memMap = {4: 5, 8: 14, 10: 13, 15: 16, 21: 22, 50: 52, 54: 55, 60: 61, 11: 14, 51: 52, 49: 51, 56: 57, 58: 59, 121: 123, 12: 13, 17: 20, 67: 68, 69: 70, 73: 74, 2: 3, 20: 22}
inputSize = 0
for k, v in memMap.items():
    for i in range(k, v + 1):
        inputSize += 1
env = gym.make('Pong-ram-v0')
nn = Net(inputSize, [10, 10], 2)
nn.initParam(0.001)

def changeColor(rgb):
    return rgb[0] * 0.299 + rgb[1] * 0.587 + rgb[2] * 0.114

def obToState(ob):
    global memMap
    ret = [0 for i in range(inputSize)]
    index = 0
    for k, v in memMap.items():
        for i in range(k, v + 1):
            ret[index] = ob[i]
            index += 1
    return ret

def getAction(res):
    return 2 if res[0] > res[1] else 3

def getOutput(state):
    x = torch.tensor(state).type(torch.FloatTensor)
    res = nn(x).tolist()
    return res

# def compareOb(ob1, ob2):
#     global memMap
#     if ob1 is None:
#         return
#     start = -1
#     for i in range(len(ob1)):
#         if start < 0:
#             if ob1[i] != ob2[i]:
#                 start = i
#         else:
#             if ob1[i] == ob2[i]:
#                 if start in memMap:
#                     memMap[start] = max(i, memMap[start])
#                 else:
#                     memMap[start] = i
#                 start = -1

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
                env.render()
                state = obToState(ob)
                obs.append(state)
                out = getOutput(state)
                act = getAction(out)
                acts.append(act)
                outputs.append(out)
                ob, reward, done, info = env.step(act)
                if reward != 0:
                    total += 1
                    # r = -10
                    # if reward > 0:
                    #     r = -r
                    r = 1
                    for k in range(len(obs)):
                        i = len(obs) - 1 - k
                        # if acts[i] == 2:
                        #     outputs[i][0] += r
                        #     outputs[i][1] -= r
                        # else:
                        #     outputs[i][0] -= r
                        #     outputs[i][1] += r
                        if reward < 0:
                            outputs[i][0], outputs[i][1] = outputs[i][1] * r, outputs[i][0] * r
                        else:
                            outputs[i][0], outputs[i][1] = outputs[i][0] * r, outputs[i][1] * r
                        if k >= 10:
                            break
                        # r *= 0.95
                        y.append(outputs[i])
                        x.append(obs[i])
                if done:
                    break
        x = torch.tensor(x).type(torch.FloatTensor)
        y = torch.tensor(y).type(torch.FloatTensor)
        nn.train(x, y, 100)
        print("finish train " + str(j))

def run(runTimes):
    count = 0
    total = 0
    while total < runTimes:
        ob = env.reset()
        for i in range(10000):
            env.render()
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
    print("win " + str(count) + " total " + str(runTimes))

train(100, 50)
print(memMap)

for i in range(3):
    run(200)


# win 6 total 200
# win 11 total 200
# win 4 total 200
# {4: 5, 8: 14, 10: 13, 15: 16, 21: 22, 50: 52, 54: 55, 60: 61, 11: 14, 51: 52, 49: 51, 56: 57, 58: 59, 121: 123, 12: 13, 17: 20, 67: 68, 69: 70, 73: 74, 2: 3, 20: 22}
# {4: 5, 8: 14, 10: 13, 15: 16, 21: 22, 50: 52, 54: 55, 60: 61, 11: 14, 51: 52, 49: 51, 56: 57, 58: 59, 121: 123, 12: 13, 17: 20, 67: 68, 69: 70, 73: 74, 2: 3, 20: 22}
# {8: 14, 10: 13, 15: 16, 21: 22, 50: 52, 54: 55, 60: 61, 11: 14, 51: 52, 4: 5, 49: 51, 56: 57, 58: 59, 121: 123, 12: 13, 17: 20, 67: 68, 69: 70, 73: 74, 2: 3, 20: 22}