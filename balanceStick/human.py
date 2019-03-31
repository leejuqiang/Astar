import gym
import torch
import time

step = 0.03
leftDown = False


def onKeyDown(key, mod):
    if key == 65361:
        global leftDown
        leftDown = True


def onKeyUp(key, mod):
    if key == 65361:
        global leftDown
        leftDown = False


def run(runTimes):
    env = gym.make('CartPole-v0')
    env.render()
    env.unwrapped.viewer.window.on_key_press = onKeyDown
    env.unwrapped.viewer.window.on_key_release = onKeyUp
    count = 0
    for ep in range(runTimes):
        ob = env.reset()
        for i in range(10000):
            env.render()
            act = 0 if leftDown else 1
            ob, reward, done, info = env.step(act)
            if done:
                count += i
                print("score " + str(i))
                break
            time.sleep(step)
    env.close()
    print(count / runTimes)


run(100)
# 21.5
# 20.72
# 20.75
