import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import seaborn as sns
import pandas as pd
import numpy as np
from gym import logger


class PGNet(nn.Module):
    def __init__(self, input_dims, fc1_dims, fc2_dims, n_actions):
        super(PGNet, self).__init__()
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, n_actions)

    def forward(self, observation):
        x = F.relu(self.fc1(observation))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out


class PGAgent(object):
    def __init__(self, lr, input_dims, n_actions, agent_name,
                 ckpt_save_path, gamma=0.99, fc1_dims=128, fc2_dims=256):
        self.reward_memory = []
        self.action_memory = []
        self.score_history = []     # episode history for plot
        self.gamma = gamma          # discount factor
        self.cur_episode = 0
        self.agent_name = f"PG_{agent_name}"
        self.ckpt_save_path = ckpt_save_path
        self.actor = PGNet(input_dims, fc1_dims, fc2_dims, n_actions)
        self.optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.actor.to(self.device)

    def __str__(self):
        return self.agent_name

    def predict(self, observation):
        x = torch.Tensor(observation).to(self.device)
        probabilities = F.softmax(self.actor.forward(x), dim=-1)
        action_probs = torch.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.action_memory.append(log_probs)

        return action.item()

    def store_rewards(self, reward):
        self.reward_memory.append(reward)

    def choose_action(self, observation):
        x = torch.Tensor(observation).to(self.device)
        _, action_t = torch.max(self.actor.forward(x), dim=-1)
        return action_t.cpu().item()

    def clear_memory(self):
        self.action_memory = []
        self.reward_memory = []

    def save_model(self, path, episode):
        torch.save({
            'model_state_dict': self.actor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'cur_episode': episode
        }, path)

    def load_model(self, path, test=False):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(
            checkpoint['optimizer_state_dict'])
        self.cur_episode = checkpoint['cur_episode']
        if test:
            self.actor.eval()
        else:
            self.actor.train()

    def plot_curve(self):
        df = pd.DataFrame(dict(episode=np.arange(len(self.score_history)),
                               score=self.score_history))
        sns_plot = sns.relplot(
            x="episode",
            y="score",
            kind="line",
            data=df)
        figure_name = os.path.join(
            self.ckpt_save_path, f"{self.agent_name}.png")
        sns_plot.savefig(figure_name)
        logger.info(f" == training figure {self.agent_name} saved")

    def learn(self):
        self.optimizer.zero_grad()

        # Calcualte discount reward G[]
        cumulate_reward = 0
        G = np.zeros_like(self.reward_memory, dtype=np.float64)
        for idx in reversed(range(len(self.reward_memory))):
            if self.reward_memory[idx] != 0:
                cumulate_reward = cumulate_reward * \
                    self.gamma + self.reward_memory[idx]
                G[idx] = cumulate_reward

        # Normalize
        mean = np.mean(G)
        std = np.std(G) if np.std(G) > 0 else 1
        G = (G - mean) / std

        loss = 0
        for g, logprob in zip(G, self.action_memory):
            loss += -g * logprob

        loss.backward()
        self.optimizer.step()

        self.clear_memory()

    def train(self, env, episodes):
        max_score = -10086
        for eps in range(self.cur_episode, episodes):
            ob = env.reset()
            score = 0
            done = False
            episode_step = 0
            while not done:
                action = self.predict(ob)
                ob, reward, done, _ = env.step(action)
                self.store_rewards(reward)
                score += reward
                episode_step += 1

            self.score_history.append(score)
            max_score = score if score > max_score else max_score
            if score > -1.0 * episode_step:
                self.learn()
                logger.info(
                    f" == episode: {eps+1}, score: {score}, max score: {max_score}")
            else:
                self.clear_memory()

            if (eps + 1) % 100 == 0:
                ckpt_name = os.path.join(
                    self.ckpt_save_path, f"ckpt_{eps}.pth")
                self.save_model(ckpt_name, eps)
                logger.info(f" == model {ckpt_name} saved")

        ckpt_name = os.path.join(self.ckpt_save_path, "ckpt_final.pth")
        self.save_model(ckpt_name, eps)
        logger.info(f" == model {ckpt_name} saved")
        self.plot_curve()

    def test(self, env):
        ob = env.reset()
        with torch.no_grad():
            score = 0
            done = False
            while not done:
                # use choose_action instead of predict
                # choose_action - choose the best action
                # predict - sample action according to probability
                action = self.choose_action(ob)
                ob, reward, done, _ = env.step(action)
                score += reward
        logger.info(f" == final score: {score}")
