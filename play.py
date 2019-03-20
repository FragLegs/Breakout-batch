import argparse
import logging

import gym
import gym.utils.play
from utils.preprocess import greyscale
from utils.wrappers import PreproWrapper, MaxAndSkipEnv

import configs


def parse_args():
    """
    Parses the arguments from the command line

    Returns
    -------
    argparse.Namespace
    """
    desc = 'Play a Gym game'
    parser = argparse.ArgumentParser(description=desc)

    env_name_help = 'The name of the environment to use'
    parser.add_argument(
        'env_name',
        type=str,
        help=env_name_help
    )

    full_help = 'Set this to play without preprocessing'
    parser.add_argument(
        '--full',
        action='store_true',
        help=full_help
    )

    verbosity_help = 'Verbosity level (default: %(default)s)'
    choices = [
        logging.getLevelName(logging.DEBUG),
        logging.getLevelName(logging.INFO),
        logging.getLevelName(logging.WARN),
        logging.getLevelName(logging.ERROR)
    ]

    parser.add_argument(
        '-v',
        '--verbosity',
        choices=choices,
        help=verbosity_help,
        default=logging.getLevelName(logging.INFO)
    )

    # Parse the command line arguments
    args = parser.parse_args()

    # Set the logging to console level
    logging.basicConfig(level=args.verbosity)

    return args


"""
To launch tensorboard, open a Terminal window and run
tensorboard --logdir=results/
Then, connect remotely to
address-ip-of-the-server:6006
6006 is the default port used by tensorboard.
"""
if __name__ == '__main__':
    args = parse_args()
    config = configs.Config(
        env_name=args.env_name,
        run_id=42,
        model_name='human',
        explore_name='human',
        batch=False
    )

    # make env
    env = gym.make(config.env_name)
    if not args.full:
        env = MaxAndSkipEnv(env, skip=config.skip_frame)
        env = PreproWrapper(
            env,
            prepro=greyscale,
            shape=(80, 80, 1),
            overwrite_render=config.overwrite_render
        )

    gym.utils.play.play(env)
