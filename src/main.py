import argparse
import sys

from src.environments import L9Env, L10Env
from src.agent import Agent


def main(flags):
    if flags.env == 9:
        env = L9Env()
    elif flags.env == 10:
        env = L10Env()
    else:
        sys.exit(1)

    agent = Agent(env=env, actions=flags.env,
                  gamma=flags.gamma, alpha=flags.alpha)

    agent.train(end_location=flags.end, episodes=flags.episodes)

    print(agent.get_optimal_route(flags.start, flags.end))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--gamma', type=float,
                        default=0.75,
                        help="The discount rate")

    parser.add_argument('--alpha', type=float,
                        default=0.9,
                        help="The learning Rate")

    parser.add_argument('--env', type=int,
                        default=9,
                        help="The environment to use, valid options are 9 or 10")

    parser.add_argument('--episodes', type=int,
                        default=1000,
                        help="The number of episodes to train with")

    parser.add_argument('--start', type=str,
                        default='L9',
                        help="The starting location on the grid")

    parser.add_argument('--end', type=str,
                        default='L1',
                        help="The ending location on the grid")

    flags, unparsed = parser.parse_known_args()

    main(flags)
