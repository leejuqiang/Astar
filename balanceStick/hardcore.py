import gym
import torch

def run(runTimes):
    env = gym.make('CartPole-v0')
    count = 0
    for ep in range(runTimes):
        ob = env.reset()
        for i in range(10000):
            env.render()
            act = 0 if ob[2] <= 0 else 1
            ob, reward, done, info = env.step(act)
            if done:
                count += i
                break
    env.close()
    print(count / runTimes)

for i in range(3):
    run(100)


# 42.17
# 42.0
# 42.27