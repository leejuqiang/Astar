# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

import logging
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
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        self.to(DEVICE)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        #self.head = nn.Linear(linear_input_size, num_classes)
        self.head = nn.Linear(1568, num_classes)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(-1, 32*7*7)
        #return self.head(x.view(x.size(0), -1))
        return self.head(x)

class Agent(object):
    def __init__(self, policy_net, target_net):
        self.policy_net = policy_net
        self.target_net = target_net
        self.update_count = 0
        self.g_loss = 0.

    """ Returns an action to be performed, either by random or prediction from the
        network
        :param model: Model representing the Q
        :param processed_observation: Current state of the game
        :return: Action to perform
    """
    def get_action(self, state, epsilon=0.5):
        rand_prob = random.random()
        if rand_prob < epsilon:
            return random.randint(UP_ACTION, DOWN_ACTION)
        state = np.expand_dims(state, 0)
        return np.argmax(self.policy_net(torch.tensor(state, dtype=torch.float).to(DEVICE)).tolist())

    def optimize(self, memory, batch_size):
        optimizer = optim.RMSprop(self.policy_net.parameters())
        loss_func = nn.MSELoss()
        for i in range(int(len(memory) / batch_size)):
            batch = memory.sample(batch_size)
            batch = list(zip(*batch)) # List of columns
            state_batch = torch.tensor(batch[0], dtype=torch.float).to(DEVICE)
            action_batch = torch.tensor(batch[1], dtype=torch.long).to(DEVICE)
            next_state_batch = torch.tensor(batch[2], dtype=torch.float).cuda()
            reward_batch = torch.tensor(batch[3], dtype=torch.float).to(DEVICE)

            a = self.target_net(next_state_batch).tolist()
            target_rewards = np.amax(a, axis=1) * GAMMA + reward_batch.tolist()
            target_rewards = torch.tensor(target_rewards, dtype=torch.float).to(DEVICE)
            policy_rewards = self.policy_net(state_batch).gather(1, action_batch)
            loss = loss_func(policy_rewards, target_rewards)
            self.g_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def _update_target_net(self):
        LOG.info("Loss at update: %.4f" % self.g_loss)
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
    num_classes = 2
    h = 64
    w = 64
    batch_size = 128
    num_episodes = 3000

    env = gym.make('Pong-v4')
    env.reset()

    #PATH = "/Users/sjjin/class/cs686/Astar/pong/weights/weight"
    PATH = "/home/jin/workspace/Astar/pong/weights/weight_add_reward"

    policy_net = PixelDQN(w, h, num_classes).cuda()
    target_net = PixelDQN(w, h, num_classes).cuda()
    target_net.load_state_dict(policy_net.state_dict())
    agent = Agent(policy_net, target_net)

    memory = ReplayMemory(300)

    train = False
    epsilon = 0.5
    frame = 1
    LOG.info("New run: total_episodes: %d", num_episodes)
    if train:
        total_rewards = [0, 0, 0, 0, 0]
        for i_episode in range(1, num_episodes+1):
            total_reward = 0
            LOG.info("i_episode: %d" % i_episode)
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
                #env.render()
                action = agent.get_action(state, epsilon=1)
                act = 2 if action == 0 else 3
                _, reward, done, _ = env.step(act)

                previous_observation = current_observation
                current_observation = get_screen(env)
                next_state = current_observation - previous_observation

                states.append(state)
                actions.append([action])
                next_states.append(next_state)
                rewards.append(reward)

                memory._push(state, [action], next_state, reward)
                if reward != 0:
                    rewards = assign_rewards(rewards, reward)
                    memory.push_all(states, actions, next_states, rewards)
                    agent.optimize(memory, batch_size)
                    states = []
                    actions = []
                    next_states = []
                    rewards = []

                    total_reward = total_reward + reward

                state = next_state
                if frame % 10000 == 0:
                    LOG.info("Frame: %d", frame)
                frame += 1

                if done:
                    #LOG.info("Episode: %d, total reward: %d" % (i_episode, total_reward))
                    next_state = None
                    epsilon = np.max([0.2, epsilon * 0.99])
                    break
            total_rewards[i_episode % 5] = total_reward
            if i_episode >= 5:
                LOG.info("Rolling mean: %.4f", np.mean(total_rewards))

            if i_episode % 5 == 0:
                torch.save(policy_net.state_dict(), PATH+str(i_episode))
                agent._update_target_net()
    else:
        env.reset()
        policy_net.load_state_dict(torch.load(PATH+str(3000)))
        while True:
            env.render()
            current_observation = get_screen(env)
            previous_observation = get_screen(env)
            state = current_observation - previous_observation
            action = agent.get_action(state, epsilon=0)

            act = 2 if action == 0 else 3
            _, reward, done, _ = env.step(act)

            if done:
                env.reset()


if __name__ == "__main__":
    main()
