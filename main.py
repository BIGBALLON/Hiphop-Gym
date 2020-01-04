import argparse
import gym
import os
from gym import wrappers, logger
from models import PGAgent, DQNAgent, test_one_episode, DDPGAgent
from atari_wrappers import wrap_deepmind

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Reinforcement Learning Algorithm for OpenAI Gym Benchmark")
    parser.add_argument('--env_name', type=str, default='CartPole-v0',
                        help='Select the environment to run')
    parser.add_argument('--output_path', type=str, default=os.getcwd(),
                        help='Output path for saving records or models')
    parser.add_argument('--mode', default='train', type=str,
                        help='Optional: [train, resume, test]')
    parser.add_argument('--env_seed', type=int, default=0,
                        help='Seed for environment')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Episode for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate for training')
    parser.add_argument('--checkpoint', default='',
                        help='Checkpoint for resume or testing')
    parser.add_argument('--save_record', action='store_true',
                        help='Save record or not')

    args = parser.parse_args()

    logger.set_level(logger.INFO)

    record_save_path = os.path.join(args.output_path, args.env_name, "records")
    ckpt_save_path = os.path.join(
        args.output_path, args.env_name, "checkpoint")
    os.makedirs(record_save_path, exist_ok=True)
    os.makedirs(ckpt_save_path, exist_ok=True)

    env = gym.make(args.env_name)
    # env = wrap_deepmind(env)

    # seed for reproducible random numbers
    if args.env_seed:
        env.seed(args.env_seed)

    if isinstance(env.action_space, gym.spaces.discrete.Discrete):
        total_actions = env.action_space.n
    elif isinstance(env.action_space, gym.spaces.box.Box):
        total_actions = env.action_space.shape[0]

    state_dims = env.observation_space.shape[0]
    action_bound = float(env.action_space.high[0])
    logger.info(f" == action space: {env.action_space}")
    logger.info(f" == action bound: {env.action_space.high}")
    logger.info(f" == observation space: {env.observation_space}")
    if args.save_record:
        env = wrappers.Monitor(
            env,
            directory=record_save_path,
            video_callable=lambda count: (count) % 50 == 0,
            force=True
        )
    agent = DDPGAgent(
        lr=args.lr,
        state_dims=state_dims,
        action_dims=total_actions,
        env_name=args.env_name,
        ckpt_save_path=ckpt_save_path,
        action_bound=action_bound,
        gamma=0.99,
        fc1_dims=128,
        fc2_dims=128
    )

    if args.mode == "resume":
        agent.load_model(args.checkpoint)
        logger.info(f" == model {args.checkpoint} loaded, continue to train")
        agent.train(env, args.episodes)
    elif args.mode == "test":
        agent.load_model(args.checkpoint, test=True)
        logger.info(f" == model {args.checkpoint} loaded, start to test")
        test_one_episode(agent, env)
    else:
        logger.info(f" == start to train from scratch")
        agent.train(env, args.episodes)

    # close the env and write monitor result info to disk
    env.close()
