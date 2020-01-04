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

MEMORY_CAPACITY = 100000
TARGET_UPDATE_ITER = 1000
BATCH_SIZE = 64
TAU = 0.001
LR_ACTOR = 0.0001          # learning rate of the actor
LR_CRITIC = 0.001          # learning rate of the critic
EPSILON_FINAL = 0.01
EPSILON_DECAY = 0.999


class Actor(nn.Module):
    def __init__(self, state_dims, action_dims, fc1_dims=256, fc2_dims=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, action_dims)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        out = torch.tanh(self.fc3(x))
        return out


class Critic(nn.Module):
    def __init__(self, state_dims, action_dims, fc1_dims=256, fc2_dims=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims + action_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, 1)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        s = F.relu((self.fc1(state)))
        x = torch.cat((s, action), dim=1)
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.4, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * \
            np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


class DDPGAgent(object):
    def __init__(self,
                 lr,
                 state_dims,
                 action_dims,
                 env_name,
                 ckpt_save_path,
                 gamma=0.99,
                 fc1_dims=512,
                 fc2_dims=512):
        self.epsilon = 1.0
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
        self.actor_eval = Actor(state_dims, action_dims)
        self.actor_target = Actor(state_dims, action_dims)
        self.critic_eval = Critic(state_dims, action_dims)
        self.critic_target = Critic(state_dims, action_dims)
        self.noise = OUNoise(action_dims, 233)

        print(self.actor_eval)
        print(self.critic_eval)

        # self.actor_eval.apply(weight_init)
        # self.critic_eval.apply(weight_init)

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.actor_eval.to(self.device)
        self.actor_target.to(self.device)
        self.critic_eval.to(self.device)
        self.critic_target.to(self.device)

        self.actor_optimizer = optim.Adam(
            self.actor_eval.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(
            self.critic_eval.parameters(), lr=LR_CRITIC, weight_decay=1e-4)
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
            action += self.noise.sample()
        action = np.clip(action, -1.0, 1.0)
        return action

    # def choose_action(self, observation):
    #     x = torch.tensor(observation).to(self.device)
    #     actions_value = self.actor_eval.forward(x)
    #     _, action = torch.max(actions_value, -1)
    #     return action.item()

    # def save_model(self, path, episode):
    #     torch.save({
    #         'model_state_dict': self.actor_eval.state_dict(),
    #         'optimizer_state_dict': self.optimizer.state_dict(),
    #         'cur_episode': episode
    #     }, path)

    # def load_model(self, path, test=False):
    #     checkpoint = torch.load(path)
    #     self.actor_eval.load_state_dict(checkpoint['model_state_dict'])
    #     self.optimizer.load_state_dict(
    #         checkpoint['optimizer_state_dict'])
    #     self.cur_episode = checkpoint['cur_episode']
    #     if test:
    #         self.actor_eval.eval()
    #     else:
    #         self.actor_eval.train()

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
                action = self.act(state)
                state_, reward, done, _ = env.step(action)
                total_step += 1
                score += reward
                reward = check_reward(
                    self.env_name, state, action, reward, state_, done
                )
                self.buffer.add(state, action, reward, done, state_)

                if self.buffer.size > BATCH_SIZE:
                    self.learn()
                state = state_

            max_score = score if score > max_score else max_score
            self.score_history.append(score)
            logger.info(
                f" == episode: {eps+1}, total step: {total_step}, score: {score}, max score: {max_score}")

            # if (eps + 1) % 100 == 0:
            #     ckpt_name = os.path.join(
            #         self.ckpt_save_path, f"ckpt_{eps}.pth")
            #     self.save_model(ckpt_name, eps)
            #     logger.info(f" == model {ckpt_name} saved")

        # ckpt_name = os.path.join(self.ckpt_save_path, "ckpt_final.pth")
        # self.save_model(ckpt_name, eps)
        # logger.info(f" == model {ckpt_name} saved")
        figure_name = os.path.join(
            self.ckpt_save_path, f"{self.agent_name}.png")
        plot_figure(figure_name, self.score_history)
