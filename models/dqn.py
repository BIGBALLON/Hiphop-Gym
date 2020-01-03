import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from gym import logger
from .utils import check_reward, plot_figure, weight_init
from .utils import ReplayBuffer

MEMORY_CAPACITY = 100000
INIT_REPLAY_SIZE = 50000
TARGET_UPDATE_ITER = 1000
BATCH_SIZE = 64
EPSILON_FINAL = 0.005
EPSILON_DECAY = 0.99


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


class DQN_Dueling_RAM(nn.Module):
    def __init__(self, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DQN_Dueling_RAM, self).__init__()
        self.fc1 = nn.Linear(input_dims, fc1_dims)

        self.fc2_adv = nn.Linear(fc1_dims, fc2_dims)
        self.fc2_val = nn.Linear(fc1_dims, fc2_dims)

        self.fc3_adv = nn.Linear(fc2_dims, n_actions)
        self.fc3_val = nn.Linear(fc2_dims, 1)

    def forward(self, observation):
        x = F.relu(self.fc1(observation))

        adv = F.relu(self.fc2_adv(x))
        val = F.relu(self.fc2_val(x))

        adv = self.fc3_adv(adv)
        val = self.fc3_val(val)

        # Q(s,a) = V(s) + A(s,a)
        # improvement:
        # Q(s,a) = V(s) + (A(s,a) - 1 / |A| sum(A(s,a')))
        adv_avg = torch.mean(adv, dim=-1, keepdim=True)
        x = val.expand_as(adv) + (adv - adv_avg)
        return x


class DQN(nn.Module):
    def __init__(self, in_channels=4, n_actions=14):
        """
        Initialize Deep Q Network
        Args:
            in_channels (int): number of input channels
            n_actions (int): number of outputs
        """
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.head = nn.Linear(512, n_actions)

    def forward(self, x):
        x = x.float() / 255
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.head(x)


class DQNAgent(object):
    def __init__(self,
                 lr,
                 input_dims,
                 n_actions,
                 env_name,
                 ckpt_save_path,
                 use_double_q=False,
                 use_dueling=True,
                 use_ram=False,
                 gamma=0.99,
                 fc1_dims=256,
                 fc2_dims=256):
        self.epsilon = 1.0
        self.gamma = gamma
        self.cur_episode = 0
        self.learn_iterations = 0
        self.buffer = ReplayBuffer(MEMORY_CAPACITY)
        self.score_history = []
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.env_name = env_name
        self.use_double_q = use_double_q
        self.agent_name = f"DQN_{env_name}"
        self.ckpt_save_path = ckpt_save_path
        if use_ram:
            if use_dueling:
                self.eval_net = DQN_Dueling_RAM(
                    input_dims, fc1_dims, fc2_dims, n_actions)
                self.target_net = DQN_Dueling_RAM(
                    input_dims, fc1_dims, fc2_dims, n_actions)
            else:
                self.eval_net = DQN_RAM(
                    input_dims, fc1_dims, fc2_dims, n_actions)
                self.target_net = DQN_RAM(
                    input_dims, fc1_dims, fc2_dims, n_actions)

        else:
            self.eval_net = DQN(4, n_actions)
            self.target_net = DQN(4, n_actions)
            print(self.target_net)
        self.eval_net.apply(weight_init)
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.target_net.to(self.device)
        self.eval_net.to(self.device)
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()

    def __str__(self):
        return self.agent_name

    def predict(self, observation):
        x = observation.to(self.device)
        if np.random.uniform() > self.epsilon:            # greedy
            actions_value = self.eval_net.forward(x)
            _, action = torch.max(actions_value, -1)
            return action.item()
        else:                                             # random
            action = np.random.randint(0, self.n_actions)
        return action

    def choose_action(self, observation):
        x = torch.Tensor(observation).to(self.device)
        actions_value = self.eval_net.forward(x)
        _, action = torch.max(actions_value, -1)
        return action.item()

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

    def learn(self):
        batch_s, batch_a, batch_r, batch_t, batch_s_ = self.buffer.sample_batch(
            BATCH_SIZE)
        self.optimizer.zero_grad()

        batch_s = torch.cat(batch_s).to(self.device)
        batch_a = torch.LongTensor(batch_a).to(self.device)
        batch_r = torch.FloatTensor(batch_r).to(self.device)
        batch_s_ = torch.cat(batch_s_).to(self.device)

        q_eval = self.eval_net(batch_s).gather(1, batch_a.view(-1, 1))
        q_next = self.target_net(batch_s_).detach()

        # use double Q
        if self.use_double_q:
            q_action = self.eval_net(batch_s_).max(1)[1].view(-1, 1)
            q_target = batch_r + self.gamma * q_next.gather(1, q_action)

        else:
            q_target = batch_r + self.gamma * q_next.max(1)[0]

        loss = self.loss_func(q_eval, q_target.view(-1, 1))

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(self.eval_net.parameters(), 1.0)
        self.optimizer.step()

        if self.epsilon > EPSILON_FINAL:
            self.epsilon = self.epsilon * EPSILON_DECAY

        # target parameter update
        if self.learn_iterations % TARGET_UPDATE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            logger.info(f" == update targe network")
        self.learn_iterations += 1

    def get_state(self, obs):
        state = np.array(obs)
        state = state.transpose((2, 0, 1))
        state = torch.from_numpy(state)
        return state.unsqueeze(0)

    def train(self, env, episodes):
        max_score = -514229
        total_step = 0
        for eps in range(self.cur_episode, episodes):
            state = env.reset()
            state = self.get_state(state)
            score = 0
            done = False
            while not done:
                action = self.predict(state)
                state_, reward, done, _ = env.step(action)
                state_ = self.get_state(state_)
                total_step += 1
                score += reward
                reward = check_reward(
                    self.env_name, state, action, reward, state_, done
                )
                self.buffer.add(state, action, reward, done, state_)

                if self.buffer.size > INIT_REPLAY_SIZE:
                    self.learn()
                elif self.buffer.size % 500 == 0:
                    print(f' == populate the replay buffer ... ... ')
                state = state_

            max_score = score if score > max_score else max_score
            self.score_history.append(score)
            logger.info(
                f" == episode: {eps+1}, total step: {total_step}, score: {score}, max score: {max_score}")

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
