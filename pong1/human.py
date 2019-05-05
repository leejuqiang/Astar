import gym
import torch
import time

isUpPressed = False
isDownPressed = False


def onKeyDown(key, mod):
    if key == 65362:
        global isUpPressed
        isUpPressed = True
    if key == 65364:
        global isDownPressed
        isDownPressed = True


def onKeyUp(key, mod):
    if key == 65362:
        global isUpPressed
        isUpPressed = False
    elif key == 65364:
        global isDownPressed
        isDownPressed = False


def getAction():
    r = random.randint(0, 1)
    return 2 if r == 0 else 3


def run(runTimes):
    env = gym.make('Pong-v0')
    env.render()
    env.unwrapped.viewer.window.on_key_press = onKeyDown
    env.unwrapped.viewer.window.on_key_release = onKeyUp
    for i in range(runTimes):
        ob = env.reset()
        done = False
        while not done:
            env.render()
            act = 0
            if isUpPressed:
                act = 2
            elif isDownPressed:
                act = 3
            ob, reward, done, info = env.step(act)
            time.sleep(0.04)
    env.close()


run(3)


# win 6 total 200
# win 6 total 200
# win 10 total 200
