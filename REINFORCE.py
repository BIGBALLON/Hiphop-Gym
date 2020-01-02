import argparse
import gym
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import seaborn as sns
import pandas as pd
import numpy as np
from gym import wrappers, logger

sns.set(style="darkgrid")


class PGActor(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(PGActor, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, observation):
        x = torch.Tensor(observation).to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out


class PGAgent(object):
    def __init__(self, lr, input_dims, n_actions, agent_name, gamma=0.99,
                 fc1_dims=128, fc2_dims=256, episode=0):
        self.reward_memory = []
        self.action_memory = []
        self.score_history = []     # episode history for plot
        self.gamma = gamma          # discount factor
        self.cur_episode = episode
        self.agent_name = f"PG_{agent_name}"
        self.actor = PGActor(lr, input_dims, fc1_dims, fc2_dims,
                             n_actions)

    def __str__(self):
        return self.agent_name

    def predict(self, observation):
        probabilities = F.softmax(self.actor.forward(observation), dim=-1)
        action_probs = torch.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.action_memory.append(log_probs)

        return action.item()

    def store_rewards(self, reward):
        self.reward_memory.append(reward)

    def clear_memory(self):
        self.action_memory = []
        self.reward_memory = []

    def choose_action(self, observation):
        _, action_t = torch.max(self.actor.forward(observation), dim=-1)
        return action_t.cpu().item()

    def save_model(self, path, episode):
        torch.save({
            'model_state_dict': self.actor.state_dict(),
            'optimizer_state_dict': self.actor.optimizer.state_dict(),
            'cur_episode': episode
        }, path)

    def load_model(self, path, test=False):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['model_state_dict'])
        self.actor.optimizer.load_state_dict(
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

        sns_plot.savefig(f"{self.agent_name}.png")
        logger.info(f" == training curve {self.agent_name} saved")

    def learn(self):
        self.actor.optimizer.zero_grad()

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
        self.actor.optimizer.step()

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
                ckpt_name = os.path.join(ckpt_save_path, f"ckpt_{eps}.pth")
                self.save_model(ckpt_name, eps)
                logger.info(f" == model {ckpt_name} saved")

        ckpt_name = os.path.join(ckpt_save_path, "ckpt_final.pth")
        self.save_model(ckpt_name, eps)
        logger.info(f" == model {ckpt_name} saved")
        self.plot_curve()

    def test(self):
        ob = env.reset()
        with torch.no_grad():
            score = 0
            done = False
            while not done:
                action = self.predict(ob)
                ob, reward, done, _ = env.step(action)
                score += reward
        logger.info(f" == final score: {score}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Reinforcement Learning Algorithm for OpenAI Gym Benchmark")
    parser.add_argument('--env_name', type=str, default='CartPole-v1',
                        help='Select the environment to run')
    parser.add_argument('--output_path', type=str, default=os.getcwd(),
                        help='Output path for saving records or models')
    parser.add_argument('--mode', default='train', type=str,
                        help='Optional: [train, resume, test]')
    parser.add_argument('--env_seed', type=int, default=0,
                        help='Seed for environment')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Episode for training')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate for training')
    parser.add_argument('--checkpoint', default='',
                        help='Checkpoint for resume or testing')
    parser.add_argument('--save_record', action='store_true',
                        help='Save record or not')

    args = parser.parse_args()

    logger.set_level(logger.INFO)

    record_save_path = os.path.join(args.output_path, "records")
    ckpt_save_path = os.path.join(args.output_path, "checkpoint")
    os.makedirs(record_save_path, exist_ok=True)
    os.makedirs(ckpt_save_path, exist_ok=True)

    env = gym.make(args.env_name)

    # seed for reproducible random numbers
    if args.env_seed:
        env.seed(args.env_seed)

    assert isinstance(
        env.action_space, gym.spaces.discrete.Discrete
    ), f"REINFORCE is only for discrete task"

    total_actions = env.action_space.n
    input_dims = env.observation_space.shape[0]
    logger.info(f" == action space: {env.action_space}")
    logger.info(f" == observation space: {env.observation_space}")
    if args.save_record:
        env = wrappers.Monitor(
            env,
            directory=record_save_path,
            video_callable=lambda count: (count) % 100 == 0,
            force=True
        )

    agent = PGAgent(
        lr=args.lr,
        input_dims=input_dims,
        n_actions=total_actions,
        agent_name=args.env_name,
        gamma=0.99,
        fc1_dims=128,
        fc2_dims=256
    )

    if args.mode == "resume":
        agent.load_model(args.checkpoint)
        logger.info(f" == model {args.checkpoint} loaded, continue to train")
        agent.train(env, args.episodes)
    elif args.mode == "test":
        agent.load_model(args.checkpoint, test=True)
        logger.info(f" == model {args.checkpoint} loaded, start to test")
        agent.test()
    else:
        logger.info(f" == start to train from scratch")
        agent.train(env, args.episodes)

    # close the env and write monitor result info to disk
    env.close()
