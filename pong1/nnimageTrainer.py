import gym
import torch
import torch.nn.functional as F
import numpy as np
import random
import copy
import cv2
import time

class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.rate = 0
        self.opt = None
        self.conv1 = torch.nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0)
        # self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        # self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        # self.bn3 = nn.BatchNorm2d(64)
        
        self.fc1 = torch.nn.Linear(7 * 7 * 64, 512)
        self.fc2 = torch.nn.Linear(512, 6)
        self.loss = torch.nn.MSELoss()

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out

    def initParam(self, rate):
        self.rate = rate
        self.opt = torch.optim.RMSprop(self.parameters(), lr=self.rate)

    def train(self, x, y, actions):
        acts = torch.tensor(actions).type(torch.LongTensor)
        out = self(x)
        out = torch.gather(out, 1, acts)
        loss = self.loss(out, y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

class Buffer:
    def __init__(self, maxSize):
        self.pool = []
        self.position = 0
        self.maxSize = maxSize

    def add(self, state, act, done, next, reward):
        # act = 0 if act == 2 else 1
        if len(self.pool) < self.maxSize:
            self.pool.append([state, [act], done, next, reward])
        else:
            self.pool[self.position] = [state, [act], done, next, reward]
            self.position += 1
            if self.position >= len(self.pool):
                self.position = 0

    def size(self):
        return len(self.pool)
        
    def sample(self, number):
        list = random.sample(self.pool, number)
        states = [list[i][0] for i in range(number)]
        actions = [list[i][1] for i in range(number)]
        dones = [list[i][2] for i in range(number)]
        nextStates = [list[i][3] for i in range(number)]
        rewards = [list[i][4] for i in range(number)]
        return states, actions, rewards, dones, nextStates

class FrameGroup:
    def __init__(self, number):
        self.frames = [None for i in range(number)]
        self.position = 0

    def addFrame(self, ob):
        self.frames[self.position] = changeOb(ob)
        self.position += 1
        if self.position >= len(self.frames):
            self.position = 0
    
    def getOb(self):
        ret = [None for i in range(len(self.frames))]
        for i in range(len(ret)):
            index = self.position + i + 1
            if index >= len(ret):
                index -= len(ret)
            ret[i] = self.frames[index]
        return ret
        
inputSize = 128
syncTime = 1000
frameG = FrameGroup(4)
# for k, v in memMap.items():
#     for i in range(k, v + 1):
#         inputSize += 1
env = gym.make('PongDeterministic-v4')
nn = Net()
nn.initParam(0.0001)
qnn = Net()
buffer = Buffer(10000)

def syncNN():
    qnn.load_state_dict(nn.state_dict())

def changeOb(ob):
    ob = ob[:, :, 0] * 0.299 + ob[:, :, 1] * 0.587 + ob[:, :, 2] * 0.114
    # ret = [[0 for j in range(160)] for i in range(210)]
    # for i in range(210):
    #     for j in range(160):
    #         ret[i][j] = changeColor(ob[i][j])
    ob = cv2.resize(ob, (84, 110), interpolation=cv2.INTER_AREA)
    ob = ob[18:102, :]
    ob = ob.astype(np.int8)
    ob = ob.astype(np.float)
    return ob

def getAction(state, rate):
    r = random.random()
    if r < rate:
        return random.randint(0, 5)
    state = np.expand_dims(state, 0)
    x = torch.tensor(state).type(torch.FloatTensor)
    res = nn(x).tolist()[0]
    return np.argmax(res)

def computeRewards(nextStates, rewards, dones):
    next = torch.tensor(nextStates).type(torch.FloatTensor)
    y = qnn(next).tolist()
    ret = [0 for i in range(len(y))]
    for i in range(len(y)):
        r = max(y[i]) if not dones[i] else 0
        ret[i] = 0.99 * r + rewards[i]
    return ret

def resetEnv():
    env.reset()
    env.step(1)
    ob, _, _, _ = env.step(2)
    for i in range(4):
        frameG.addFrame(ob)
    return frameG.getOb()

def trainOnce():
    states, actions, rewards, dones, nextStates = buffer.sample(32)
    rs = computeRewards(nextStates, rewards, dones)
    rs = np.reshape(rs, [32, 1])
    # rs = [[0, 0]]
    x = torch.tensor(states).type(torch.FloatTensor)
    y = torch.tensor(rs).type(torch.FloatTensor)
    nn.train(x, y, actions)

def train(epoch):
    st = syncTime
    ob = resetEnv()
    nextState = ob
    for i in range(100):
        state = nextState
        env.render()
        act = getAction(state, 1)
        lastOb = ob
        ob, reward, done, info = env.step(act)
        frameG.addFrame(ob)
        nextState = frameG.getOb()
        buffer.add(state, act, done, nextState, reward)
    print("start to train")
    randomRate = 0.8
    for j in range(epoch):
        count = 0
        total = 0
        done = False
        ob = resetEnv()
        nextState = ob
        while not done:
            state = nextState
            env.render()
            act = getAction(state, randomRate)
            ob, reward, done, info = env.step(act)
            frameG.addFrame(ob)
            nextState = frameG.getOb()
            buffer.add(state, act, done, nextState, reward)
            trainOnce()
            if randomRate > 0.02:
                randomRate -= 0.00001
            st -= 1
            if st <= 0:
                print("sync net")
                syncNN()
                st = syncTime
            if reward != 0:
                total += reward
        print("finish train " + " epoch " + str(j) + " reward " + str(total))
        if j % 10 == 0:
            torch.save(nn.state_dict(), "save/nn_" + str(j) + ".txt")
    print("finish all train")

def run(times):
    for i in range(times):
        ob = resetEnv()
        nextState = ob
        done = False
        while not done:
            state = nextState
            env.render()
            act = getAction(state, 0)
            lastOb = ob
            ob, reward, done, info = env.step(act)
            frameG.addFrame(ob)
            nextState = frameG.getOb()
            time.sleep(0.03)

# train(200)
# torch.save(nn.state_dict(), "save/nn.txt")

nn.load_state_dict(torch.load("save/nn_90.txt"))
run(3)

# win 6 total 200
# win 11 total 200
# win 4 total 200
# {4: 5, 8: 14, 10: 13, 15: 16, 21: 22, 50: 52, 54: 55, 60: 61, 11: 14, 51: 52, 49: 51, 56: 57, 58: 59, 121: 123, 12: 13, 17: 20, 67: 68, 69: 70, 73: 74, 2: 3, 20: 22}
# {4: 5, 8: 14, 10: 13, 15: 16, 21: 22, 50: 52, 54: 55, 60: 61, 11: 14, 51: 52, 49: 51, 56: 57, 58: 59, 121: 123, 12: 13, 17: 20, 67: 68, 69: 70, 73: 74, 2: 3, 20: 22}
# {8: 14, 10: 13, 15: 16, 21: 22, 50: 52, 54: 55, 60: 61, 11: 14, 51: 52, 4: 5, 49: 51, 56: 57, 58: 59, 121: 123, 12: 13, 17: 20, 67: 68, 69: 70, 73: 74, 2: 3, 20: 22}
