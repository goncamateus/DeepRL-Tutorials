import hfo
import numpy as np
from scipy.spatial import distance


class HFOEnv(hfo.HFOEnvironment):
    pitchHalfLength = 52.5
    pitchHalfWidth = 34
    tolerance_x = pitchHalfLength * 0.1
    tolerance_y = pitchHalfWidth * 0.1
    FEAT_MIN = -1
    FEAT_MAX = 1
    pi = 3.14159265358979323846
    max_R = np.sqrt(pitchHalfLength * pitchHalfLength +
                    pitchHalfWidth * pitchHalfWidth)
    stamina_max = 8000

    def __init__(self, actions, rewards, play_goalie=False):
        super(HFOEnv, self).__init__()
        self.connectToServer(hfo.HIGH_LEVEL_FEATURE_SET, './formations-dt',
                             6000, 'localhost', 'base_right', play_goalie=play_goalie)

        class ObservationSpace():
            def __init__(self, env, rewards):
                self.state = env.getState()
                self.rewards = rewards
                self.nmr_out = 0
                self.taken = 0
                self.shape = self.state.shape

        class ActionSpace():
            def __init__(self, actions):
                self.actions = actions
                self.n = len(actions)

        self.observation_space = ObservationSpace(self, rewards)
        self.action_space = ActionSpace(actions)
        self.num_teammates = self.getNumTeammates()
        self.num_opponents = self.getNumOpponents()

    def step(self, action, act_args=None, is_offensive=False):
        action = self.action_space.actions[action]
        if act_args: self.act(action, act_args)
        else: self.act(action)
        act = self.action_space.actions.index(action)
        status = super(HFOEnv, self).step()
        if status != hfo.IN_GAME:
            done = True
        else:
            done = False
        next_state = self.get_state(is_offensive)
        # -----------------------------
        if is_offensive:
            reward = self.get_reward_off(act, next_state, done, status)
        else:
            reward = self.get_reward_def(act, next_state, done, status)
        return next_state, reward, done, status

    def get_state(self, is_offensive=False):
        state = self.remake_state(self.getState(), is_offensive=is_offensive)
        return state

    def get_reward_off(self, act, next_state, done, status):
        reward = 0
        return reward

    def get_reward_def(self, act, next_state, done, status):
        reward = 0
        if status == hfo.GOAL:
            reward = -20000
        elif not '-{}'.format(self.getUnum()) in self.statusToString(status):
            reward = 0
        elif 'OUT' in self.statusToString(status):
            self.observation_space.nmr_out += 1
            reward = self.observation_space.rewards[act]/2
            if self.observation_space.nmr_out % 20 and self.observation_space.nmr_out > 1:
                reward = reward*10
        else:
            if done:
                reward = self.observation_space.rewards[act]
                if '-{}'.format(self.getUnum()) in self.statusToString(status):
                    self.observation_space.taken += 1
                    reward = self.observation_space.rewards[act]*2
                    if self.observation_space.taken % 5 and self.observation_space.taken > 1:
                        reward = reward*20
            else:
                reward = self.observation_space.rewards[act] - next_state[3]*3

        return reward

    def unnormalize(self, val, min_val, max_val):
        return ((val - self.FEAT_MIN) / (self.FEAT_MAX - self.FEAT_MIN)) * (max_val - min_val) + min_val

    def abs_x(self, normalized_x_pos, playing_offense):
        if playing_offense:
            return self.unnormalize(normalized_x_pos, -self.tolerance_x, self.pitchHalfLength + self.tolerance_x)
        else:
            return self.unnormalize(normalized_x_pos, -self.pitchHalfLength-self.tolerance_x, self.tolerance_x)

    def abs_y(self, normalized_y_pos):
        return self.unnormalize(normalized_y_pos, -self.pitchHalfWidth - self.tolerance_y,
                                self.pitchHalfWidth + self.tolerance_y)

    def remake_state(self, state, is_offensive=False):
        num_mates, num_ops = self.num_teammates, self.num_opponents
        state[0] = self.abs_x(state[0], is_offensive)
        state[1] = self.abs_y(state[1])
        state[2] = self.unnormalize(state[2], -self.pi, self.pi)
        state[3] = self.abs_x(state[3], is_offensive)
        state[4] = self.abs_y(state[4])
        state[5] = self.unnormalize(state[5], 0, 1)
        state[6] = self.unnormalize(state[6], 0, self.max_R)
        state[7] = self.unnormalize(state[7], -self.pi, self.pi)
        state[8] = self.unnormalize(state[8], 0, self.pi)
        if num_ops > 0:
            state[9] = self.unnormalize(state[9], 0, self.max_R)
        else:
            state[9] = -1000
        for i in range(10, 10+num_mates):
            if state[i] != -2:
                state[i] = self.unnormalize(state[i], 0, self.pi)
            else:
                state[i] = -1000
        for i in range(10 + num_mates, 10 + 2*num_mates):
            if state[i] != -2:
                state[i] = self.unnormalize(state[i], 0, self.max_R)
            else:
                state[i] = -1000
        for i in range(10 + 2*num_mates, 10 + 3*num_mates):
            if state[i] != -2:
                state[i] = self.unnormalize(state[i], 0, self.pi)
            else:
                state[i] = -1000
        index = 10 + 3*num_mates
        for i in range(num_mates):
            if state[index] != -2:
                state[index] = self.abs_x(state[index], is_offensive)
            else:
                state[index] = -1000
            index += 1
            if state[index] != -2:
                state[index] = self.abs_y(state[index])
            else:
                state[index] = -1000
            index += 2
        index = 10 + 6*num_mates
        for i in range(num_ops):
            if state[index] != -2:
                state[index] = self.abs_x(state[index], is_offensive)
            else:
                state[index] = -1000
            index += 1
            if state[index] != -2:
                state[index] = self.abs_y(state[index])
            else:
                state[index] = -1000
            index += 2
        state[-1] = self.unnormalize(state[-1], 0, self.stamina_max)
        return state

    def get_dist(self, v1, v2):
        return distance.euclidean(v1, v2)

    def get_ball_dist(self, state):
        agent = (state[0], state[1])
        ball = (state[3], state[4])
        return distance.euclidean(agent, ball)

    def strict_state(self, state, choosed_mates, choosed_ops, is_offensive=False):
        num_mates, num_ops = self.num_teammates, self.num_opponents
        new_state = state[:10].tolist()
        for i in range(10, 10+choosed_mates):
            new_state.append(state[i])
        for i in range(10 + num_mates, 10 + num_mates + choosed_mates):
            new_state.append(state[i])
        for i in range(10 + 2*num_mates, 10 + 2*num_mates + choosed_mates):
            new_state.append(state[i])
        index = 10 + 3*num_mates
        for i in range(choosed_mates):
            new_state.append(state[index])
            index += 1
            new_state.append(state[index])
            index += 1
            new_state.append(state[index])
            index += 1
        index = 10 + 6*num_mates
        for i in range(choosed_ops):
            new_state.append(state[index])
            index += 1
            new_state.append(state[index])
            index += 1
            new_state.append(state[index])
            index += 1
        new_state.append(state[-2])
        new_state.append(state[-1])
        new_state = np.array(new_state)
        return new_state
        
