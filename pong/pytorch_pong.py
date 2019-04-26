# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable


#is_ipython = 'inline' in matplotlib.get_backend()
#if is_ipython:
#    from IPython import display
#
#plt.ion()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Transition = namedtuple("Transition",
                       ("state", "action", "next_state", "reward"))


class GameStateData(Dataset):
    def __init__(self, observations, rewards):
        #super(GameStateData).__init__()
        self.observations = observations
        self.rewards = rewards

    def __getitem__(self, index):
        observation = torch.tensor(self.observations[index], dtype=torch.float32)
        reward = torch.tensor(self.rewards[index], dtype=torch.int64)
        #observation = self.observations[index]
        #reward = self.rewards[index]
        return observation, reward

    def __len__(self):
        return len(self.observations)

class PongDQN(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(PongDQN, self).__init__()

        self.linear1 = nn.Linear(input_size, 200)
        self.linear2 = nn.Linear(200, 10)
        self.linear3 = nn.Linear(10, num_classes)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        #x = self.conv1(x)
        #x = self.bn1(x)
        #x = x.view(-1)
        #x = F.relu(x)
        #x = F.relu(self.bn1(self.conv1(x)))
        #x = F.relu(self.bn2(self.conv2(x)))
        #x = F.relu(self.bn3(self.conv3(x)))
        return x


Transition = namedtuple("Transition",
                       ("state", "action", "next_state", "reward"))

def train(dataloader, model):
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.MultiLabelSoftMarginLoss()

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(1):
        running_loss = 0.0
        for i, (data, label) in enumerate(dataloader):
            optimizer.zero_grad()
            data = Variable(data.view(-1, 128))
            label = Variable(label)

            #output = model(data.view(128, 1, 3))
            output = model(data)
            #import pdb; pdb.set_trace()
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()


UP_ACTION = 2
DOWN_ACTION = 3

def main():
    seed = 42
    env = gym.make('Pong-ram-v4')
    observation = env.reset()

    input_size = observation.shape[0]
    num_classes = 3

    model = PongDQN(input_size, num_classes)

    observations = []
    rewards = []
    count = 0
    while True:
        #env.render()
        action = random.randint(UP_ACTION, DOWN_ACTION)
        observation, reward, done, info = env.step(action)
        observations.append(observation)
        rewards.append(np.sign(reward))
        if count == 10 or done:
            #import pdb; pdb.set_trace()
            rewards = np.zeros(shape=(1,11)) #debugging
            data = GameStateData(observations, rewards)
            dataloader = DataLoader(data)


            train(dataloader, model)
            observations = []
            rewards = []
        count += 1




main()

# env = gym.make("Pong-v0")
# observation = env.reset()
# env.render()
# prev_processed_observations = None
# processed_observations, prev_processed_observations =\
#     preprocess_observations(observation, prev_processed_observations,
#                             W, H)
# print(processed_observations.shape)
# t = torch.from_numpy(processed_observations)
# print(t.shape)
# print("HelloWorld")
