import gym
import torch
import random
import json


def toByte(n, min, max):
    n = (n - min) / (max - min)
    if n > 1:
        n = 0.99
    if n < 0:
        n = 0
    n *= 4
    return int(n)


def obToState(ob):
    ret = 0
    ret |= toByte(ob[0], -2.4, 2.4)
    ret |= toByte(ob[1], -3.0, 3.0) << 2
    ret |= toByte(ob[2], -0.5, 0.5) << 4
    ret |= toByte(ob[3], -2.0, 2.0) << 6
    return ret


table = [[0 for i in range(256)], [0 for i in range(256)]]


def saveToFile(t, num):
    s = json.dumps(t)
    f = open("save/bs_" + str(num) + ".json", "w")
    f.write(s)
    f.close()


def loadFromFile(path):
    f = open("save/" + path, "r")
    s = f.read()
    return json.loads(s)


def getAct(state, useGreedy):
    r = random.random()
    if useGreedy:
        global greedy
        if r < greedy:
            return env.action_space.sample()
        greedy *= 0.9
    r0 = table[0][state]
    r1 = table[1][state]
    return 0 if r0 > r1 else 1


alpha = 0.2
gamma = 0.9
greedy = 0.2


def updateTable(state, nextState, act, reward):
    r = table[act][state]
    nextAct = getAct(nextState, False)
    table[act][state] = (1 - alpha) * r + alpha * \
        (reward + gamma * table[nextAct][nextState])


env = gym.make('CartPole-v0')


def score():
    count = 0
    for ep in range(50):
        ob = env.reset()
        for i in range(10000):
            env.render()
            s = obToState(ob)
            act = getAct(s, False)
            ob, reward, done, info = env.step(act)
            if done:
                count += i
                break
    return count / 50


def train(runTimes):
    states = []
    acts = []
    obs = []
    maxS = 0
    maxTable = None
    for ep in range(runTimes):
        ob = env.reset()
        for i in range(10000):
            env.render()
            s = obToState(ob)
            act = getAct(s, True)
            states.append(s)
            acts.append(act)
            obs.append(ob)
            ob, reward, done, info = env.step(act)
            if done:
                rewards = [-10 for i in range(len(states))]
                lastAng = obs[len(obs) - 1][2]
                invert = True
                f = 1
                for i in range(len(obs) - 1, -1, -1):
                    if not invert and ((obs[i][2] < 0 and lastAng > 0) or (obs[i][2] > 0 and lastAng < 0)):
                        invert = False
                    if not invert:
                        rewards[i] = 10
                    else:
                        rewards[i] = -10
                    rewards[i] *= f
                    f *= 0.9
                for i in range(len(states) - 1):
                    updateTable(states[i], states[i + 1], acts[i], rewards[i])
                s = score()
                if s > maxS:
                    maxS = s
                    maxTable = []
                    maxTable.append(table[0].copy())
                    maxTable.append(table[1].copy())
                break
    saveToFile(maxTable, runTimes)
    print("finish training")
    return maxTable


def run(runTimes):
    count = 0
    for ep in range(runTimes):
        ob = env.reset()
        for i in range(10000):
            env.render()
            act = getAct(obToState(ob), False)
            ob, reward, done, info = env.step(act)
            if done:
                count += i
                break
    env.close()
    print(count / runTimes)


# table = train(200)

table = loadFromFile("bs_200.json")
for i in range(1):
    run(100)

# 157.17
# 166.1
# 157.68
