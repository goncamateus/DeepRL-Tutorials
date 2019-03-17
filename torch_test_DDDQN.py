import glob
import itertools
import math
import os
from datetime import timedelta
from timeit import default_timer as timer

import hfo
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from scipy.spatial import distance

from agents.DQN import Model as DQN_Agent
from hfo_env import HFOEnv
from utils.hyperparameters import Config
from utils.plot import plot_reward
from utils.ReplayMemory import ExperienceReplayMemory

config = Config()

config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# epsilon variables
config.epsilon_start = 1.0
config.epsilon_final = 0.01
config.epsilon_decay = 30000
config.epsilon_by_frame = lambda frame_idx: config.epsilon_final + \
    (config.epsilon_start - config.epsilon_final) * \
    math.exp(-1. * frame_idx / config.epsilon_decay)

# misc agent variables
config.GAMMA = 0.99
config.LR = 1e-4
# memory
config.TARGET_NET_UPDATE_FREQ = 1000
config.EXP_REPLAY_SIZE = 100000
config.BATCH_SIZE = 16
config.PRIORITY_ALPHA=0.6
config.PRIORITY_BETA_START=0.4
config.PRIORITY_BETA_FRAMES = 100000

#epsilon variables
config.SIGMA_INIT=0.5

#Learning control variables
config.LEARN_START = 1000
config.MAX_FRAMES=50000000
config.UPDATE_FREQ = 1

# Nstep controls
config.N_STEPS = 1


class DuelingDQN(nn.Module):
    def __init__(self, input_shape, num_outputs):
        super(DuelingDQN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_outputs

        self.mlp1 = nn.Linear(self.input_shape[0], 32)
        self.mlp2 = nn.Linear(32, 64)
        self.mlp3 = nn.Linear(64, 64)

        self.adv1 = nn.Linear(self.feature_size(), 512)
        self.adv2 = nn.Linear(512, self.num_actions)

        self.val1 = nn.Linear(self.feature_size(), 512)
        self.val2 = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.mlp1(x))
        x = F.relu(self.mlp2(x))
        x = F.relu(self.mlp3(x))
        x = x.view(x.size(0), -1)

        adv = F.relu(self.adv1(x))
        adv = self.adv2(adv)

        val = F.relu(self.val1(x))
        val = self.val2(val)

        return val + adv - adv.mean()

    def feature_size(self):
        x = self.mlp1(torch.zeros(1, self.input_shape[0]))
        x = self.mlp2(x)
        x = self.mlp3(x)
        return x.view(1, -1).size(1)

    def sample_noise(self):
        # ignore this for now
        pass


class Model(DQN_Agent):
    def __init__(self, static_policy=False, env=None, config=None):
        super(Model, self).__init__(static_policy, env, config)

    def declare_networks(self):
        self.model = DuelingDQN(
            self.env.observation_space.shape, self.env.action_space.n)
        self.target_model = DuelingDQN(
            self.env.observation_space.shape, self.env.action_space.n)
        
actions = [hfo.MOVE, hfo.GO_TO_BALL]
rewards = [700, 1000]
hfo_env = HFOEnv(actions, rewards)

start = timer()
log_dir = "/tmp/gym/"
try:
    os.makedirs(log_dir)
except OSError:
    files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)

model = Model(env=hfo_env, config=config)

model_path = './saved_agents/model_{}.dump'.format(hfo_env.getUnum())
optim_path = './saved_agents/optim_{}.dump'.format(hfo_env.getUnum())
mem_path = './saved_agents/exp_replay_agent_{}.dump'.format(hfo_env.getUnum())

if os.path.isfile(model_path) and os.path.isfile(optim_path):
    model.load_w(model_path=model_path, optim_path=optim_path)
    print("Model Loaded")

if os.path.isfile(mem_path):
    model.load_replay(mem_path=mem_path)
    config.LEARN_START = 0
    print("Memory Loaded")

max_reached = False
frame_idx = 1

for episode in itertools.count():
    status = hfo.IN_GAME
    episode_rewards = []
    if max_reached:
        break
    while status == hfo.IN_GAME and not max_reached:
        state = hfo_env.get_state()
        epsilon = config.epsilon_by_frame(frame_idx)
        action = model.get_action(state, epsilon)
        if hfo_env.get_ball_dist(state) > 20:
           action = 0

        next_state, reward, done, status = hfo_env.step(action)
        episode_rewards.append(reward)
        model.update(state, action, reward, next_state, frame_idx)

        if done:
            # Get the total reward of the episode
            total_reward = np.sum(episode_rewards)
            model.finish_nstep()
            model.reset_hx()
            # We finished the episode
            next_state = np.zeros(state.shape)
        else:
            state = next_state
        frame_idx += 1
        if frame_idx == config.MAX_FRAMES+1:
            max_reached = True
            break
        if frame_idx%10000 == 0:
            model.save_w(path_model='./saved_agents/model_{}.dump'.format(hfo_env.getUnum()),
                        path_optim='./saved_agents/optim_{}.dump'.format(hfo_env.getUnum()))
            print("Model Saved")
#------------------------------ DOWN
# Quit if the server goes down
    if status == hfo.SERVER_DOWN or max_reached:
        model.save_w(path_model='./saved_agents/model_{}.dump'.format(hfo_env.getUnum()),
                    path_optim='./saved_agents/optim_{}.dump'.format(hfo_env.getUnum()))
        print("Model Saved")
        model.save_replay(mem_path=mem_path)
        print("Memory Saved")
        hfo_env.act(hfo.QUIT)
        exit()
