# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

import gym
import math
import random

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.autograd import Variable
from utils.util import preprocess_observations
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UP_ACTION = 2
DOWN_ACTION = 3

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

class PixelDQN(torch.nn.Module):
    def __init__(self, num_classes):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, num_classes)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

def train(dataloader, model):
    criterion = nn.CrossEntropyLoss()
    #criterion = nn.MultiLabelSoftMarginLoss()

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(1):
        running_loss = 0.0
        for i, (data, label) in enumerate(dataloader):
            optimizer.zero_grad()
            data = Variable(data.view(-1, 128))
            label = Variable(label)

            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

""" Returns an action to be performed, either by random or prediction from the
    network
    :param model: Model representing the Q
    :param processed_observation: Current state of the game
    :param random_explore: Probability to return a random move
    :return: Action to perform
"""
def get_action(model, processed_observation, random_explore=0.2):
    rand_prob = random.random()
    if rand_prob < random_explore:
        return random.randint(UP_ACTION, DOWN_ACTION)
    else:
        return model(processed_observation)

""" Assigns the reward to individual observations in a given sequence of observations in
    one episode
    :param rewards: List of observations (processed pixels) for one episode (scoring event)
    :param label: True reward of the sequence of events
    :return: Rewards with discounted factor
"""
def assign_reward(rewards, label, gamma=0.99):
    rewards = [r * gamma**(len(rewards)-i) for i, x in enumerate(rewards)]
    return rewards


def main():
    seed = 42
    env = gym.make('Pong-v4')
    observation = env.reset()
    prev_processed_observation = None

    num_classes = 2
    h = 64
    w = 64

    model = PixelDQN(num_classes)

    episode_observations = []
    episode_rewards = []
    number_of_games_left = 1
    while number_of_games_left > 0:
        env.render()
        processed_observations, prev_processed_observations =\
                preprocess_observations(observation, prev_processed_observations,
                                        h, w)

        action = get_action(model, processed_observation, random_explore)
        observation, reward, done, info = env.step(action)
        observations.append(observation)
        rewards.append(np.sign(reward))
        # End of an episode (a scoring event happend)
        if reward != 0:
            degraded_rewards = assign_reward(rewards)
            episode_rewards = episode_rewards + degraded_rewards
            episode_observations = episode_observations + observations
            observations = []
            rewards = []
            if done:
                data = GameStateData(episode_observations, episode_rewards)
                dataloader = DataLoader(data)

                train(dataloader, model)
                # clear the observations and rewards
                episode_observations = []
                episode_rewards = []
                number_of_games_left -= 1

if __name__ == "__main__":
    main()
