import gym
import torch
import random


def getAction():
    r = random.randint(0, 1)
    return 2 if r == 0 else 3


def run(runTimes):
    env = gym.make('Pong-ram-v0')
    count = 0
    total = 0
    while total < runTimes:
        ob = env.reset()
        for i in range(10000):
            env.render()
            act = env.action_space.sample()
            ob, reward, done, info = env.step(act)
            if reward != 0:
                total += 1
                if reward > 0:
                    count += 1
            if done:
                break
    env.close()
    print("win " + str(count) + " total " + str(runTimes))


for i in range(3):
    run(200)


# win 6 total 200
# win 6 total 200
# win 10 total 200
