
import torch
import seaborn as sns
import pandas as pd
import numpy as np
from gym import logger

sns.set(style="darkgrid")


def check_reward(env, state, action, reward, state_, done, episode_step):
    if("CartPole" in env):
        return trick_for_cartpole(done, reward)
    elif("MountainCar-v0" in env):
        return trick_for_mountaincar(state, done, reward, state_, episode_step)


def trick_for_cartpole(done, reward):
    """
    trick for speed up cartpole training
    if done, which means agent died, set negtive reward,
    which help agent learn control method faster.
    """
    if done:
        return -100
    return reward


def trick_for_mountaincar(state, done, reward, state_, episode_step):
    """
    -1 for each time step, until the goal position of 0.5 is reached. 
    As with MountainCarContinuous v0, there is no penalty for climbing the left hill, 
    which upon reached acts as a wall.
    state[0] means position:  -1.2 ~ 0.6
    state[1] velocity:  -0.07 ~ 0.07
    """
    return abs(state_[1])


def test_one_episode(agent, env):
    ob = env.reset()
    with torch.no_grad():
        score = 0
        done = False
        while not done:
            # use choose_action instead of predict
            # choose_action - choose the best action
            # predict - sample action according to probability
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
