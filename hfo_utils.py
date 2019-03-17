import numpy as np
from scipy.spatial import distance

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


def unnormalize(val, min_val, max_val):
    return ((val - FEAT_MIN) / (FEAT_MAX - FEAT_MIN)) * (max_val - min_val) + min_val


def abs_x(normalized_x_pos, playing_offense):
    if playing_offense:
        return unnormalize(normalized_x_pos, -tolerance_x, pitchHalfLength + tolerance_x)
    else:
        return unnormalize(normalized_x_pos, -pitchHalfLength-tolerance_x, tolerance_x)


def abs_y(normalized_y_pos):
    return unnormalize(normalized_y_pos, -pitchHalfWidth - tolerance_y,
                       pitchHalfWidth + tolerance_y)


def remake_state(state, num_mates, num_ops, is_offensive=False):
    state[0] = abs_x(state[0], is_offensive)
    state[1] = abs_y(state[1])
    state[2] = unnormalize(state[2], -pi, pi)
    state[3] = abs_x(state[3], is_offensive)
    state[4] = abs_y(state[4])
    state[5] = unnormalize(state[5], 0, 1)
    state[6] = unnormalize(state[6], 0, max_R)
    state[7] = unnormalize(state[7], -pi, pi)
    state[8] = unnormalize(state[8], 0, pi)
    if num_ops > 0:
        state[9] = unnormalize(state[9], 0, max_R)
    else:
        state[9] = -1000
    for i in range(10, 10+num_mates):
        if state[i] != -2:
            state[i] = unnormalize(state[i], 0, pi)
        else:
            state[i] = -1000
    for i in range(10 + num_mates, 10 + 2*num_mates):
        if state[i] != -2:
            state[i] = unnormalize(state[i], 0, max_R)
        else:
            state[i] = -1000
    for i in range(10 + 2*num_mates, 10 + 3*num_mates):
        if state[i] != -2:
            state[i] = unnormalize(state[i], 0, pi)
        else:
            state[i] = -1000
    index = 10 + 3*num_mates
    for i in range(num_mates):
        if state[index] != -2:
            state[index] = abs_x(state[index], is_offensive)
        else:
            state[index] = -1000
        index += 1
        if state[index] != -2:
            state[index] = abs_y(state[index])
        else:
            state[index] = -1000
        index += 2
    index = 10 + 6*num_mates
    for i in range(num_ops):
        if state[index] != -2:
            state[index] = abs_x(state[index], is_offensive)
        else:
            state[index] = -1000
        index += 1
        if state[index] != -2:
            state[index] = abs_y(state[index])
        else:
            state[index] = -1000
        index += 2
    state[-1] = unnormalize(state[-1], 0, stamina_max)
    return state


def get_dist(v1, v2):
    return distance.euclidean(v1, v2)


def strict_state(state, choosed_mates, choosed_ops, num_mates, num_ops, is_offensive=False):
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
