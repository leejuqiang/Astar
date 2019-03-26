import gym
import torch
import time
import pyHook

def run(runTimes):
    env = gym.make('CartPole-v0')
    count = 0
    for ep in range(runTimes):
        ob = env.reset()
        for i in range(10000):
            env.render()
            act = env.action_space.sample()
            ob, reward, done, info = env.step(act)
            if done:
                count += i
                break
            else:
                time.sleep(0.03)
    env.close()
    print(count / runTimes)

def onKeyDown(event):
    print(event.key)

hm = pyHook.HookManager()
hm.KeyDown = onKeyDown
hm.HookKeyboard()

run(10)
# 21.5
# 20.72
# 20.75