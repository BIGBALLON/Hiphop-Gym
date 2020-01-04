
import torch
import random
import torch.nn as nn
import seaborn as sns
import pandas as pd
import numpy as np
import pickle
from gym import logger

sns.set(style="darkgrid")


class ReplayBuffer(object):
    def __init__(self, max_size):
        self.max_size = max_size
        self.size = 0
        self.buffer = []

    def add(self, s, a, r, t, s_):
        experience = (s, a, r, t, s_)
        if self.size < self.max_size:
            self.buffer.append(experience)
            self.size += 1
        else:
            self.buffer.pop(0)
            self.buffer.append(experience)

    def len(self):
        return self.size

    def sample_batch(self, batch_size):
        if self.size < batch_size:
            batch = random.sample(self.buffer, self.size)
        else:
            batch = random.sample(self.buffer, batch_size)

        batch_s = [_[0] for _ in batch]
        batch_a = [_[1] for _ in batch]
        batch_r = [_[2] for _ in batch]
        batch_t = [_[3] for _ in batch]
        batch_s_ = [_[4] for _ in batch]

        return batch_s, batch_a, batch_r, batch_t, batch_s_

    def clear(self):
        self.buffer = []
        self.size = 0

    def save(self):
        file = open('replay_buffer.pkl', 'wb')
        pickle.dump(self.buffer, file)
        file.close()

    def load(self):
        try:
            filehandler = open('replay_buffer.pkl', 'rb')
            self.buffer = pickle.load(filehandler)
            self.size = len(self.buffer)
        except:
            print(f"no file is loaded")


def discount_reward(reward_memory, gamma):
    cumulate_reward = 0
    G = np.zeros_like(reward_memory)
    for idx in reversed(range(len(reward_memory))):
        if reward_memory[idx] != 0:
            cumulate_reward = cumulate_reward * \
                gamma + reward_memory[idx]
            G[idx] = cumulate_reward
    return G


def weight_init(m):
    """
    orthogonal initialize for better performance
    """
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        # nn.init.orthogonal_(m.weight)
        nn.init.kaiming_normal_(m.weight.data)


def check_reward(env, state, action, reward, state_, done):
    if("CartPole" in env):
        return trick_for_cartpole(done, reward)
    elif("MountainCar-v0" in env):
        return trick_for_mountaincar(state, done, reward, state_)
    elif("Acrobot-v1" in env):
        return trick_for_acrobot(state, done, reward, state_)
    else:
        return reward


def trick_for_cartpole(done, reward):
    """
    trick for speed up cartpole training
    if done, which means agent died, set negtive reward,
    which help agent learn control method faster.
    """
    if done:
        return -100
    return reward


def trick_for_mountaincar(state, done, reward, state_):
    """
    -1 for each time step, until the goal position of 0.5 is reached. 
    As with MountainCarContinuous v0, there is no penalty for climbing the left hill, 
    which upon reached acts as a wall.
    state[0] means position:  -1.2 ~ 0.6
    state[1] velocity:  -0.07 ~ 0.07
    """
    return abs(state_[1])


def trick_for_acrobot(state, done, reward, state_):
    """
    TBD
    """
    rewardx = (-np.cos(state_[0]) - np.cos(state_[1] + state_[0]))
    if rewardx < .5:
        reward = -1.
    if (rewardx > .5 and rewardx < .8):
        reward = -0.8
    if rewardx > .8:
        reward = -0.6
    if rewardx > 1:
        reward = -0.4
    return reward


def get_state(obs):
    # state = np.array(obs)
    # state = state.transpose((2, 0, 1))
    # state = torch.from_numpy(state)
    # return state.unsqueeze(0)
    return obs


def test_one_episode(agent, env):
    ob = env.reset()
    with torch.no_grad():
        score = 0
        done = False
        while not done:
            # use choose_action instead of predict
            # choose_action - choose the best action
            # predict - sample action according to probability
            ob = get_state(ob)
            action = agent.choose_action(ob)
            ob, reward, done, _ = env.step(action)
            score += reward
    logger.info(f" == final score: {score}")


def plot_figure(path, score_history):
    df = pd.DataFrame(dict(episode=np.arange(len(score_history)),
                           score=score_history))
    sns_plot = sns.relplot(
        x="episode",
        y="score",
        kind="line",
        data=df)
    sns_plot.savefig(path)
    logger.info(f" == training figure {path} saved")
