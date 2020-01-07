import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import copy
from gym import logger
from .utils import check_reward, plot_figure, weight_init
from .utils import ReplayBuffer

MEMORY_CAPACITY = 500000
MIN_STEP_TO_TRAIN = 5000
BATCH_SIZE = 64
TAU = 0.001
LR_ACTOR = 0.0005          # learning rate of the actor
LR_CRITIC = 0.001          # learning rate of the critic


class OUNoise:
    def __init__(self, n_actions, mu=0, theta=0.15, sigma=0.2):
        self.n_actions = n_actions
        self.X = np.ones(n_actions) * mu
        self.mu = mu
        self.sigma = sigma
        self.theta = theta

    def reset(self):
        self.X = np.ones(self.n_actions) * self.mu

    def sample(self):
        dX = self.theta * (self.mu - self.X)
        dX += self.sigma * np.random.randn(self.n_actions)
        self.X += dX
        return self.X


class Actor(nn.Module):
    def __init__(self, state_dims, action_dims, action_bound, fc1_dims=400, fc2_dims=300):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, action_dims)

        self.action_bound = action_bound

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        out = self.action_bound * torch.tanh(self.fc3(x))
        return out


class Critic(nn.Module):
    def __init__(self, state_dims, action_dims, fc1_dims=400, fc2_dims=300):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dims + action_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, 1)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = F.relu(self.fc1(torch.cat([state, action], 1)))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out


class DDPGAgent(object):
    def __init__(self,
                 lr,
                 state_dims,
                 action_dims,
                 env_name,
                 ckpt_save_path,
                 action_bound,
                 gamma=0.99,
                 fc1_dims=400,
                 fc2_dims=300):
        self.gamma = gamma
        self.cur_episode = 0
        self.learn_iterations = 0
        self.buffer = ReplayBuffer(MEMORY_CAPACITY)
        self.score_history = []
        self.action_dims = action_dims
        self.state_dims = state_dims
        self.env_name = env_name
        self.agent_name = f"DDPG_{env_name}"
        self.ckpt_save_path = ckpt_save_path
        self.actor_eval = Actor(state_dims, action_dims, action_bound)
        self.actor_target = copy.deepcopy(self.actor_eval)

        self.critic_eval = Critic(state_dims, action_dims)
        self.critic_target = copy.deepcopy(self.critic_eval)
        self.action_bound = action_bound
        self.noise = OUNoise(self.action_dims)

        print(self.actor_eval)
        print(self.critic_eval)

        self.actor_eval.apply(weight_init)
        self.critic_eval.apply(weight_init)

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.actor_eval.to(self.device)
        self.actor_target.to(self.device)
        self.critic_eval.to(self.device)
        self.critic_target.to(self.device)

        self.actor_optimizer = optim.Adam(
            self.actor_eval.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(
            self.critic_eval.parameters(), lr=LR_CRITIC)
        self.actor_loss = nn.MSELoss()
        self.critic_loss = nn.MSELoss()

    def __str__(self):
        return self.agent_name

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.tensor(state).float().to(self.device)
        self.actor_eval.eval()
        with torch.no_grad():
            action = self.actor_eval(state).cpu().data.numpy()
        self.actor_eval.train()

        if add_noise:
            action += self.action_bound * self.noise.sample()

        action = np.clip(action, -self.action_bound, self.action_bound)
        return action

    def choose_action(self, state):
        state = torch.tensor(state).float().to(self.device)
        self.actor_eval.eval()
        with torch.no_grad():
            action = self.actor_eval(state).cpu().data.numpy()
        return action

    def save_model(self, path, episode):
        torch.save({
            'actor_model_state_dict': self.actor_eval.state_dict(),
            'critic_model_state_dict': self.critic_eval.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'cur_episode': episode
        }, path)

    def load_model(self, path, test=False):
        checkpoint = torch.load(path)
        self.actor_eval.load_state_dict(checkpoint['actor_model_state_dict'])
        self.critic_eval.load_state_dict(checkpoint['critic_model_state_dict'])
        self.actor_optimizer.load_state_dict(
            checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(
            checkpoint['critic_optimizer_state_dict'])
        self.cur_episode = checkpoint['cur_episode']
        if test:
            self.actor_eval.eval()
        else:
            self.actor_eval.train()
        self.actor_target.load_state_dict(self.actor_eval.state_dict())
        self.critic_target.load_state_dict(self.critic_eval.state_dict())

    def learn(self):
        batch_s, batch_a, batch_r, batch_t, batch_s_ = self.buffer.sample_batch(
            BATCH_SIZE)

        batch_s = torch.Tensor(batch_s).to(self.device)
        batch_a = torch.Tensor(batch_a).to(self.device)
        batch_r = torch.Tensor(batch_r).to(self.device).view((-1, 1))
        batch_t = torch.Tensor(batch_t).to(self.device).view((-1, 1))
        batch_s_ = torch.Tensor(batch_s_).to(self.device)

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(batch_s_)
        Q_targets_next = self.critic_target(batch_s_, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = batch_r + (self.gamma * Q_targets_next * (1 - batch_t))
        # Compute critic loss
        Q_expected = self.critic_eval(batch_s, batch_a)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_eval(batch_s)
        actor_loss = -self.critic_eval(batch_s, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_eval, self.critic_target, TAU)
        self.soft_update(self.actor_eval, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                tau*local_param.data + (1.0-tau)*target_param.data)

    def train(self, env, episodes):
        max_score = -514229
        total_step = 0
        for eps in range(self.cur_episode, episodes):
            state = env.reset()
            score = 0
            done = False

            while not done:
                if total_step < MIN_STEP_TO_TRAIN:
                    action = env.action_space.sample()
                else:
                    action = self.act(state)

                state_, reward, done, _ = env.step(action)
                total_step += 1
                score += reward
                reward = check_reward(
                    self.env_name, state, action, reward, state_, done
                )
                self.buffer.add(state, action, reward, done, state_)

                if self.buffer.size > MIN_STEP_TO_TRAIN:
                    self.learn()

                state = state_

            max_score = score if score > max_score else max_score
            self.score_history.append(score)
            logger.info(
                f" == episode: {eps+1:05d} | total step: {total_step:7d} | score: {score:8.2f} | max score: {max_score:8.2f}")

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
