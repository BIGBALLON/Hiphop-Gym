
import torch
import seaborn as sns
import pandas as pd
import numpy as np
from gym import logger


def check_reward(env, done, reward):
    if('CartPole' in env):
        return trick_for_cartpole(done, reward)


def trick_for_cartpole(done, reward):
    """
    trick for speed up cartpole training
    if done, which means agent died, set negtive reward,
    which help agent learn control method faster.
    """
    if done:
        return -100
    return reward


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
