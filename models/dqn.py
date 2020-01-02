import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import seaborn as sns
import pandas as pd
import numpy as np
from gym import logger

MEMORY_CAPACITY = 2000
INIT_REPLAY_SIZE = 1000
TARGET_UPDATE_ITER = 200
BATCH_SIZE = 64


class DQN_RAM(nn.Module):
    def __init__(self, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DQN_RAM, self).__init__()
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, n_actions)

    def forward(self, observation):
        x = F.relu(self.fc1(observation))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out


class DQNAgent(object):
    def __init__(self, lr, input_dims, n_actions, agent_name,
                 ckpt_save_path, gamma=0.99, fc1_dims=128, fc2_dims=256):
        self.epsilon = 1.0
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999
        self.gamma = gamma
        # for target updating
        self.learn_iterations = 0
        # for storing memory
        self.memory_counter = 0
        # initialize memory
        self.memory = np.zeros((MEMORY_CAPACITY, input_dims * 2 + 2))
        self.score_history = []
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.cur_episode = 0
        self.agent_name = f"DQN_{agent_name}"
        self.ckpt_save_path = ckpt_save_path
        self.eval_net = DQN_RAM(input_dims, fc1_dims, fc2_dims, n_actions)
        self.target_net = DQN_RAM(input_dims, fc1_dims, fc2_dims, n_actions)
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.target_net.to(self.device)
        self.eval_net.to(self.device)
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()

    def __str__(self):
        return self.agent_name

    def predict(self, observation):
        x = torch.Tensor(observation).to(self.device)
        if np.random.uniform() > self.epsilon:            # greedy
            actions_value = self.eval_net.forward(x)
            _, action = torch.max(actions_value, -1)
            return action.item()
        else:                                             # random
            action = np.random.randint(0, self.n_actions)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, r, s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        pass

    def save_model(self, path, episode):
        torch.save({
            'model_state_dict': self.eval_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'cur_episode': episode
        }, path)

    def load_model(self, path, test=False):
        checkpoint = torch.load(path)
        self.eval_net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(
            checkpoint['optimizer_state_dict'])
        self.cur_episode = checkpoint['cur_episode']
        if test:
            self.eval_net.eval()
        else:
            self.eval_net.train()

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
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]

        b_s = torch.FloatTensor(b_memory[:, :self.input_dims]).to(self.device)
        b_a = torch.LongTensor(
            b_memory[:, self.input_dims:self.input_dims+1].astype(int)).to(self.device)
        b_r = torch.FloatTensor(
            b_memory[:, self.input_dims+1:self.input_dims+2]).to(self.device)
        b_s_ = torch.FloatTensor(
            b_memory[:, -self.input_dims:]).to(self.device)

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()

        #  =================== double DQN ===================
        # q_action = self.eval_net(b_s_).max(1)[1].view(BATCH_SIZE, 1)
        # q_target = b_r + GAMMA * q_next.gather(1, q_action).view(BATCH_SIZE, 1)   # shape (batch, 1)
        #  =================== double DQN ===================

        #  =================== DQN ===================
        q_target = b_r + self.gamma * q_next.max(1)[0].view(BATCH_SIZE, 1)
        #  =================== DQN ===================
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.epsilon_decay

        # target parameter update
        if self.learn_iterations % TARGET_UPDATE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_iterations += 1

    def train(self, env, episodes):
        max_score = -10086
        for eps in range(self.cur_episode, episodes):
            state = env.reset()
            score = 0
            done = False
            episode_step = 0
            while not done:
                action = self.predict(state)
                state_, reward, done, _ = env.step(action)
                score += reward

                # if done:  # trick for speed up training
                #     reward = -100

                self.store_transition(state, action, reward,  state_)
                if self.memory_counter > INIT_REPLAY_SIZE:
                    self.learn()
                elif self.memory_counter % 100 == 0:
                    print(
                        f'Populate the replay buffer: {float(self.memory_counter*100)/INIT_REPLAY_SIZE:.2f}%')
                state = state_

            episode_step += 1
            max_score = score if score > max_score else max_score
            self.score_history.append(score)
            logger.info(
                f" == episode: {eps+1}, score: {score}, max score: {max_score}")

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
