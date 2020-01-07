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
MIN_STEP_TO_TRAIN = 10000
BATCH_SIZE = 128
TAU = 0.005
LR_ACTOR = 0.0001          # learning rate of the actor
LR_CRITIC = 0.0001         # learning rate of the critic


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3Agent(object):
    def __init__(self,
                 lr,
                 state_dims,
                 action_dims,
                 env_name,
                 ckpt_save_path,
                 action_bound,
                 gamma=0.99,
                 fc1_dims=256,
                 fc2_dims=256,
                 policy_noise=0.2,
                 noise_clip=0.5,
                 policy_freq=2):
        self.gamma = gamma
        self.cur_episode = 0
        self.learn_iterations = 0
        self.buffer = ReplayBuffer(MEMORY_CAPACITY)
        self.score_history = []
        self.action_dims = action_dims
        self.state_dims = state_dims
        self.env_name = env_name
        self.agent_name = f"TD3_{env_name}"
        self.ckpt_save_path = ckpt_save_path
        self.actor_eval = Actor(state_dims, action_dims, action_bound)
        self.actor_target = copy.deepcopy(self.actor_eval)

        self.critic_eval = Critic(state_dims, action_dims)
        self.critic_target = copy.deepcopy(self.critic_eval)
        self.action_bound = action_bound

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

        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0

    def __str__(self):
        return self.agent_name

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.tensor(state).float().to(self.device)
        action = self.actor_eval(state).cpu().data.numpy()
        if add_noise:
            # action += self.noise.sample()
            action += np.random.normal(0, self.action_bound *
                                       0.1, size=self.action_dims)

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

        self.total_it += 1

        batch_s = torch.Tensor(batch_s).to(self.device)
        batch_a = torch.Tensor(batch_a).to(self.device)
        batch_r = torch.Tensor(batch_r).to(self.device).view((-1, 1))
        batch_t = torch.Tensor(batch_t).to(self.device).view((-1, 1))
        batch_s_ = torch.Tensor(batch_s_).to(self.device)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(batch_a) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                self.actor_target(batch_s_) + noise
            ).clamp(-self.action_bound, self.action_bound)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(batch_s_, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = batch_r + batch_t * self.gamma * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic_eval(batch_s, batch_a)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + \
            F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            actor_loss = - \
                self.critic_eval.Q1(batch_s, self.actor_eval(batch_s)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
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
