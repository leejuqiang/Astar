# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

import math
import random
import gym

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.autograd import Variable
from utils.image_processing import preprocess_observation
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UP_ACTION = 2
DOWN_ACTION = 3
GAMMA = 0.99

class ReplayMemory(object):
    def __init__(self, memory_capacity):
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def push_all(self, list_of_state_action_rewards):
        for data in list_of_state_action_rewards):
            self._push(data[0], data[1], data[2], data[3])

    def _push(self, state, action, next_state, reward):
        data = [state, action, next_state, reward]
        if len(self.memory) < self.memory_capacity:
            self.memory.append(data)
        self.memory[self.position] = data
        self.position = (self.position + 1) % self.memory_capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

class PixelDQN(torch.nn.Module):
    def __init__(self, w, h, num_classes):
        super(PixelDQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        self.optimizer = optim.RMSprop(self.parameters(), lr=ALPHA)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, num_classes)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, observation):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

class Agent(object):
    def __init__(self, policy_net, target_net):
        self.policy_net = policy_net
        self.target_net = target_net
        self.update_count = 0

    """ Returns an action to be performed, either by random or prediction from the
        network
        :param model: Model representing the Q
        :param processed_observation: Current state of the game
        :param random_explore: Probability to return a random move
        :return: Action to perform
    """
    def get_action(self, state, epsilon=0.5, random_exploration=0.2):
        rand_prob = random.random()
        if random_explore == 1 or rand_prob < epsilon:
            return random.randint(UP_ACTION, DOWN_ACTION)
        return self.policy_net(state)

    def optimize(self, memory):
        optimizer = optim.RMSProp(self.pociy_net.parameters())
        for i, batch in enumerate(len(memory) / BATCH_SIZE):
            batch = memory.sample(BATCH_SIZE)
            state_batch = torch.tensor(batch[:, 0], dtype=torch.int8)
            action_batch = torch.tensor(batch[:, 1], dtype=torch.int8)
            next_state_batch = torch.tensor(batch[:, 2], dtype=torch.int8)
            reward_batch = torch.tensor(batch[:, 3], dtype=torch.float)
            q_values = []
            for s in state_batch:
                q_values.append()

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
        if update_count % TARGET_UPDATE == 0:
            self._update_policy_net()

    def _update_policy_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

""" Assigns the reward to individual observations in a given sequence of observations in
    one episode
    :param rewards: List of observations (processed pixels) for one episode (scoring event)
    :param label: True reward of the sequence of events
    :return: Rewards with discounted factor
"""
def assign_reward(rewards, label):
    rewards = [label * GAMMA**(len(rewards)-i) for i, x in enumerate(rewards)]
    return rewards

def main():
    num_classes = 2
    h = 64
    w = 64

    seed = 42
    env = gym.make('Pong-v4')
    env.reset()

    model = PixelDQN(w, h, num_classes)
    memory = ReplayMemory(64)

    num_episodes = 10
    episode_start_frame = 0
    episode_end_frame = 0
    for i_episode in range(num_episodes):
        states = []
        actions = []
        next_states = []
        rewards = []

        current_observation = get_screen()
        previous_observation = get_screent()
        state = current_observation - previous_observation
        while True:
            episode_end_frame += 1
            action = get_action(state)
            _, reward, done, _ = env.step(action)

            previous_observation = current_observation
            current_observation = get_screen()
            next_state = current_observation - previous_observation
            # Assign reward according to label
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            rewards.append(reward)
            episode_states.append([state, action, next_state, reward])

            # Match has ended
            if reward != 0:
                reward = np.sign(reward) # cap the score between -1 and 1
                rewards = assign_rewards(rewards)
                memory.add_all(zip(states, actions, next_states, rewards)
            state = next_state

            if done:
                next_state = None
                episode_states = []
                episode_start_frame = 0
                episode_end_frame = 0
                break

        if i_episode % TRAIN_EPISODE == 0:
            optimize(memory)
            memory = ReplayMemory(64)

if __name__ == "__main__":
    main()
