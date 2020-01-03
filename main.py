import argparse
import gym
import os
from gym import wrappers, logger
from models import PGAgent, DQNAgent, test_one_episode

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
    parser.add_argument('--lr', type=float, default=5e-4,
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
            # force=True
        )

    agent = PGAgent(
        lr=args.lr,
        input_dims=input_dims,
        n_actions=total_actions,
        env_name=args.env_name,
        ckpt_save_path=ckpt_save_path,
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
        test_one_episode(agent, env)
    else:
        logger.info(f" == start to train from scratch")
        agent.train(env, args.episodes)

    # close the env and write monitor result info to disk
    env.close()
