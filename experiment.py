import argparse
import logging
import os
import pprint
import shutil

import gym
import tensorflow
import sys
import wandb

from utils.preprocess import greyscale
from utils.wrappers import PreproWrapper, MaxAndSkipEnv

import core.nature
import core.counting_dqn

import configs
from exploration import EXPLORATION_MAP
import schedule

# make wandb tensorboard work
sys.modules['tensorflow.summary'] = tensorflow.summary


MODEL_MAP = {
    'DQN': core.nature.NatureQN,
    'CountingDQN': core.counting_dqn.CountingDQN
}


def parse_args():
    """
    Parses the arguments from the command line

    Returns
    -------
    argparse.Namespace
    """
    desc = 'Run an experiment in the cs234 project'
    parser = argparse.ArgumentParser(description=desc)

    explore_name_help = 'The name of the exploration strategy to use'
    parser.add_argument(
        'explore_name',
        type=str,
        help=explore_name_help
    )

    env_name_help = 'The name of the environment to use'
    parser.add_argument(
        '--env-name',
        type=str,
        default='Breakout-v0',
        help=env_name_help
    )

    batch_help = 'Set this flag to train on batch'
    parser.add_argument(
        '--batch',
        action='store_true',
        help=batch_help
    )

    test_help = 'Set this to test on a small sample size'
    parser.add_argument(
        '--test',
        action='store_true',
        help=test_help
    )

    run_id_help = 'Which run is this'
    parser.add_argument(
        '--run-id',
        type=str,
        default=0,
        help=run_id_help
    )

    force_help = 'Force a run even if folder exists'
    parser.add_argument(
        '--force',
        action='store_true',
        help=force_help
    )

    alt_name_help = 'Alternative name for the run'
    parser.add_argument(
        '--alt-name',
        type=str,
        default=None,
        help=alt_name_help
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
    pprint.pprint(args.__dict__)

    config = configs.Config(**args.__dict__)

    if args.force and os.path.exists(config.output_path):
        shutil.rmtree(config.output_path)

    if os.path.exists(config.output_path) and not args.test:
        raise ValueError('{} already exists!'.format(config.output_path))

    if args.test:
        reduction = 1000
        config.saving_freq //= reduction
        config.nsteps_train //= reduction
        config.buffer_size //= reduction
        config.target_update_freq //= reduction
        config.eps_nsteps //= reduction
        config.learning_start //= reduction
        config.num_episodes_test = 5

    # Uncomment this section to report results to Weights and Biases.
    # Requires that you have set up an account and entered the relevant
    # values in your environment variables.

    # wandb.init(
    #     project='cs234-project',
    #     tensorboard=True,
    #     # tensorboardX=False,
    #     dir=config.output_path
    # )
    # wandb.config.update(config.__dict__)
    # wandb.config.run_id = args.run_id
    # wandb.config.test = args.test

    # make env
    env = gym.make(config.env_name)
    env = MaxAndSkipEnv(env, skip=config.skip_frame)
    env = PreproWrapper(
        env,
        prepro=greyscale,
        shape=(80, 80, 1),
        overwrite_render=config.overwrite_render
    )

    # exploration strategy
    exp_schedule = EXPLORATION_MAP[config.explore](
        env, config.eps_begin, config.eps_end, config.eps_nsteps
    )

    # learning rate schedule
    lr_schedule  = schedule.LinearSchedule(
        config.lr_begin, config.lr_end, config.lr_nsteps
    )

    # train model
    model = MODEL_MAP[config.model](env, config)
    model.run(exp_schedule, lr_schedule)
