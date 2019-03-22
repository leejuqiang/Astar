import gym


class Stick:
    def __init__(self):
        self.env = gym.make('CartPole-v0')

    def getAct(self):
        return self.env.action_space.sample()

    def stepData(self, ob, reward):
        pass

    def train(self):
        count = 0
        for ep in range(100):
            ob = self.env.reset()
            for i in range(1000):
                self.env.render()
                act = self.getAct()
                ob, reward, done, info = self.env.step(act)
                self.stepData(ob, reward)
                if done:
                    count += i
                    break
        self.env.close()
        print(count / 100)


class StickHardCore(Stick):

    lastAng = 0

    def getAct(self):
        if self.lastAng <= 0:
            return 0
        else:
            return 1

    def stepData(self, ob, reward):
        self.lastAng = ob[2]

class StickNN(Stick):

    def getAct(self):
        return 0
    
    def stepData(self, ob, reward):
        pass

# print("random:")
# stickRandom = Stick()
# for i in range(3):
#     stickRandom.train()
# random:
# 22.39
# 22.22
# 22.49

# print("hard core:")
# stickHard = StickHardCore()
# for i in range(3):
#     stickHard.train()
# hard core:
# 43.08
# 41.34
# 41.79