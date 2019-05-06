# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

import logging
import random
import gym

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.autograd import Variable
from utils.image_processing import get_screen
from torch.utils.data import Dataset, DataLoader

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

# create a file handler
handler = logging.FileHandler('weights/test.log')
handler.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create a logging format
formatter = logging.Formatter('%(asctime)s - %(message)s')
handler.setFormatter(formatter)

# add the handlers to the logger
LOG.addHandler(handler)
LOG.addHandler(ch)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#DEVICE = "cpu"
ALPHA = 0.0001
UP_ACTION = 0
DOWN_ACTION = 1
GAMMA = 0.99
TRAIN_EPISODE = 5

class ReplayMemory(object):
    def __init__(self, memory_capacity):
        self.memory = []
        self.memory_capacity = memory_capacity
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def push_all(self, states, actions, next_states, rewards):
        for i in range(len(states)):
            self._push(states[i], actions[i], next_states[i], rewards[i])

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
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.linear = nn.Linear(64*7*7, 512)

        self.loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=0.0001)

        self.to(DEVICE)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        #self.head = nn.Linear(linear_input_size, num_classes)
        self.head = nn.Linear(512, num_classes)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(-1, 64*7*7)
        x = F.relu(self.linear(x))
        #return self.head(x.view(x.size(0), -1))
        return self.head(x)

class Agent(object):
    def __init__(self, policy_net, target_net):
        self.policy_net = policy_net
        self.target_net = target_net
        self.update_count = 0
        self.g_loss = 0.
        self.sync = 1

    def get_action(self, state, epsilon=0.5):
        rand_prob = random.random()
        if rand_prob < epsilon:
            return random.randint(0, 5)
        state = np.expand_dims(state, 0)
        return np.argmax(self.policy_net(torch.tensor(state, dtype=torch.float).to(DEVICE)).tolist())

    def optimize(self, memory, batch_size, every_frame=True):
        num_train = 1
        if not every_frame:
            num_train = int(1000 / batch_size)

        for i in range(num_train):
            if self.sync % 1000 == 0:
                self._update_target_net()

            state_batch, action_batch, next_state_batch, reward_batch =\
                    self.get_batch(memory, batch_size)

            q_targets = self.calculate_q_target(next_state_batch, reward_batch)
            q_policy = self.policy_net(state_batch).gather(1, action_batch)

            loss = self.policy_net.loss(q_policy, q_targets)
            self.g_loss += loss
            self.policy_net.optimizer.zero_grad()
            loss.backward()
            self.policy_net.optimizer.step()
            self.sync += 1

    def get_batch(self, memory, batch_size):
        batch = memory.sample(batch_size)
        batch = list(zip(*batch)) # List of columns

        state_batch = torch.tensor(batch[0], dtype=torch.float).to(DEVICE)
        action_batch = torch.tensor(batch[1], dtype=torch.long).to(DEVICE)
        next_state_batch = torch.tensor(batch[2], dtype=torch.float).cuda()
        reward_batch = torch.tensor(batch[3], dtype=torch.float).to(DEVICE)

        return state_batch, action_batch, next_state_batch, reward_batch

    def calculate_q_target(self, next_state_batch, reward_batch):
        q_targets = self.target_net(next_state_batch).tolist()
        q_targets = np.amax(q_targets, axis=1) * GAMMA + reward_batch.tolist()
        return torch.tensor(q_targets, dtype=torch.float).to(DEVICE)

    def _update_target_net(self):
        if (self.sync+1) % 10000 == 0:
            LOG.info("Loss update at frame: %d,  %.4f" % (self.sync, self.g_loss))
        self.g_loss = 0.
        self.target_net.load_state_dict(self.policy_net.state_dict())

""" Assigns the reward to individual observations in a given sequence of observations in
    one episode
    :param rewards: List of observations (processed pixels) for one episode (scoring event)
    :param label: True reward of the sequence of events
    :return: Rewards with discounted factor
"""
def assign_rewards(rewards, label):
    rewards = [label * 0.99**(len(rewards)-i) for i, x in enumerate(rewards)]
    #rewards = [label] * len(rewards)
    return rewards

def main():
    env = gym.make('PongDeterministic-v4')
    env.reset()
    num_classes = 6
    h = 80
    w = 80
    batch_size = 32
    num_episodes = 5
    epsilon = 1
    memory_size = 10000
    train_every_frame = True
    assign_reward = False
    min_frames = 100

    policy_net = PixelDQN(w, h, num_classes).cuda()
    target_net = PixelDQN(w, h, num_classes).cuda()
    target_net.load_state_dict(policy_net.state_dict())
    agent = Agent(policy_net, target_net)

    memory = ReplayMemory(memory_size)

    train = True
    name = "trial1_train_every_step_"
    #PATH = "/Users/sjjin/class/cs686/Astar/pong/weights/weight"
    PATH = "/home/jin/workspace/Astar/pong/weights/" + name
    frame = 1
    LOG.info("New run: total_episodes: %d", num_episodes)
    if train:
        rewards_window = [0, 0, 0, 0, 0]
        total_reward = 0.

        for i_episode in range(num_episodes):
            episode_reward = 0
            states = []
            actions = []
            next_states = []
            rewards = []

            env.reset()
            #env.render()
            current_observation = get_screen(env)
            previous_observation = get_screen(env)
            state = current_observation
            while True:
                env.render()
                action = agent.get_action(state, epsilon=epsilon)
                _, reward, done, _ = env.step(action)

                previous_observation = current_observation
                current_observation = get_screen(env)
                next_state = current_observation - previous_observation

                state = next_state
                frame += 1
                if train_every_frame:
                    memory._push(state, [action], next_state, reward)
                    if frame > min_frames:
                        agent.optimize(memory, batch_size)
                else:
                    states.append(state)
                    actions.append([action])
                    next_states.append(next_state)
                    rewards.append(reward)
                    if reward != 0:
                        if assign_reward:
                            rewards = assign_rewards(rewards, reward)
                        memory.push_all(states, actions, next_states, rewards)
                        if frame > min_frames:
                            agent.optimize(memory, batch_size)
                        rewards = []
                        states = []
                        actions = []
                        next_states = []
                if reward != 0:
                    episode_reward += reward

                if done:
                    rewards_window[i_episode % 5] = episode_reward
                    total_reward += episode_reward
                    #LOG.info("Episode: %d, total_mean: %.6f, rolling_mean: %.4f" % (i_episode,
                    #          total_reward / (i_episode+1), np.mean(rewards_window)))
                    LOG.info("Episode: %d, total_mean: %.6f" % (i_episode,
                              total_reward / (i_episode+1)))
                    next_state = None
                    epsilon = np.max([0.2, epsilon * 0.995])
                    break
            if (i_episode+1) % 5 == 0:
                print("Saving!")
                torch.save(policy_net.state_dict(), PATH+str(i_episode))
    else:
        env.reset()
        agent.policy_net.load_state_dict(torch.load(PATH))
        runs = 3
        while runs > 0:
            env.render()
            current_observation = get_screen(env)
            previous_observation = get_screen(env)
            state = current_observation - previous_observation
            action = agent.get_action(state, epsilon=0)

            act = 2 if action == 0 else 3
            _, reward, done, _ = env.step(act)

            if done:
                env.reset()
                runs -= 1


if __name__ == "__main__":
    main()
