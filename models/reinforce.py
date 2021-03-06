import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from gym import logger
from .utils import plot_figure, check_reward, discount_reward, weight_init


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
    def __init__(self, lr, input_dims, n_actions, env_name,
                 ckpt_save_path, gamma=0.99, fc1_dims=128, fc2_dims=256):
        self.reward_memory = []
        self.action_memory = []
        self.score_history = []     # episode history for plot
        self.gamma = gamma          # discount factor
        self.cur_episode = 0
        self.env_name = env_name
        self.agent_name = f"PG_{env_name}"
        self.ckpt_save_path = ckpt_save_path
        self.actor = PGNet(input_dims, fc1_dims, fc2_dims, n_actions)
        self.actor.apply(weight_init)
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
        _, action = torch.max(self.actor.forward(x), dim=-1)
        return action.item()

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

    def learn(self):
        self.optimizer.zero_grad()

        # Calcualte discount reward G[]
        G = discount_reward(self.reward_memory, self.gamma)

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
        max_score = -514229
        total_step = 0
        for eps in range(self.cur_episode, episodes):
            state = env.reset()
            score = 0
            done = False
            episode_step = 0
            while not done:
                action = self.predict(state)
                state_, reward, done, _ = env.step(action)
                episode_step += 1
                total_step += 1
                score += reward
                reward = check_reward(
                    self.env_name, state, action, reward, state_, done
                )
                self.store_rewards(reward)
                state = state_

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
        figure_name = os.path.join(
            self.ckpt_save_path, f"{self.agent_name}.png")
        plot_figure(figure_name, self.score_history)
