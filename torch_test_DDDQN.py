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
from hfo_utils import remake_state, strict_state
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
        
    

def get_ball_dist(state):
    agent = (state[0], state[1])
    ball = (state[3], state[4])
    return distance.euclidean(agent, ball)


class HFOEnv():
    def __init__(self, shape, actions):
        class ObservationSpace():
            def __init__(self, shape):
                self.shape = shape

        class ActionSpace():
            def __init__(self, n):
                self.n = n
        self.observation_space = ObservationSpace(shape)
        self.action_space = ActionSpace(actions)


hfo_env = hfo.HFOEnvironment()
hfo_env.connectToServer(hfo.HIGH_LEVEL_FEATURE_SET, './formations-dt',
                        6000, 'localhost', 'base_right', play_goalie=False)
num_teammates = hfo_env.getNumTeammates()
num_opponents = hfo_env.getNumOpponents()
actions = [hfo.MOVE, hfo.GO_TO_BALL]
rewards = [700, 1000]

start = timer()

log_dir = "/tmp/gym/"
try:
    os.makedirs(log_dir)
except OSError:
    files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)

model = Model(env=HFOEnv((10 + 6*num_teammates + 3 *
                          num_opponents + 2, ), 2), config=config)

path_model = './saved_agents/model_{}.dump'.format(hfo_env.getUnum())
path_optim = './saved_agents/optim_{}.dump'.format(hfo_env.getUnum())
if os.path.isfile(path_model) and os.path.isfile(path_optim):
    print("Model Loaded")
    model.load_w(model_path=path_model, optim_path=path_optim)

max_reached = False
frame_idx = 1

for episode in itertools.count():
    status = hfo.IN_GAME
    episode_rewards = []
    if max_reached:
        break
    while status == hfo.IN_GAME and not max_reached:
        state = hfo_env.getState()
        state = remake_state(
            state, num_teammates, num_opponents, is_offensive=False)

        epsilon = config.epsilon_by_frame(frame_idx)
        action = model.get_action(state, epsilon)

        if get_ball_dist(state) < 20:
            hfo_env.act(actions[action])

            act = action
        else:
            act = 0
            action = 0
            hfo_env.act(actions[0])
        # ------------------------------
        status = hfo_env.step()
        if status != hfo.IN_GAME:
            done = 1
        else:
            done = 0
        next_state = hfo_env.getState()
        next_state = remake_state(
            next_state, num_teammates, num_opponents, is_offensive=False)
        # -----------------------------
        reward = 0
        if status == hfo.GOAL:
            reward = -20000
        elif not '-{}'.format(hfo_env.getUnum()) in hfo_env.statusToString(status):
            reward = 0
        elif 'OUT' in hfo_env.statusToString(status):
            nmr_out += 1
            reward = rewards[act]/2
            if nmr_out % 20 and nmr_out > 1:
                reward = reward*10
        else:
            if done:
                reward = rewards[act]
                if '-{}'.format(hfo_env.getUnum()) in hfo_env.statusToString(status):
                    taken += 1
                    reward = rewards[act]*2
                    if taken % 5 and taken > 1:
                        reward = reward*20
            else:
                reward = rewards[act] - next_state[3]*3
        episode_rewards.append(reward)
        # -----------------------------------
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
    hfo_env.act(hfo.QUIT)
    exit()
