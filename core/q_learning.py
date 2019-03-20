from collections import deque
import functools
import os
import sys

import gym
import numpy as np

from utils.general import get_logger, Progbar, export_plot
from utils.replay_buffer import ReplayBuffer
from utils.preprocess import greyscale
from utils.wrappers import PreproWrapper, MaxAndSkipEnv


class QN(object):
    """
    Abstract Class for implementing a Q Network
    """

    def __init__(self, env, config, logger=None):
        """
        Initialize Q Network and env

        Args:
            config: class with hyperparameters
            logger: logger instance from logging module
        """
        # directory for training outputs
        if not os.path.exists(config.output_path):
            os.makedirs(config.output_path)

        # store hyper params
        self.config = config
        self.logger = logger
        if logger is None:
            self.logger = get_logger(config.log_path)
        self.env = env

        # build model
        self.build()

    def build(self):
        """
        Build model
        """
        pass

    @property
    def policy(self):
        """
        model.policy(state) = action
        """
        return lambda state: self.get_action(state)

    def save(self):
        """
        Save model parameters

        Args:
            model_path: (string) directory
        """
        pass

    def initialize(self):
        """
        Initialize variables if necessary
        """
        pass

    def get_best_action(self, state):
        """
        Returns best action according to the network

        Args:
            state: observation from gym
        Returns:
            tuple: action, q values
        """
        raise NotImplementedError

    def get_action(self, state):
        """
        Returns action with some epsilon strategy

        Args:
            state: observation from gym
        """
        if np.random.random() < self.config.soft_epsilon:
            return self.env.action_space.sample()
        else:
            return self.get_best_action(state)[0]

    def update_target_params(self):
        """
        Update params of Q' with params of Q
        """
        pass

    def init_averages(self):
        """
        Defines extra attributes for tensorboard
        """
        self.avg_reward = -21.
        self.max_reward = -21.
        self.std_reward = 0

        self.avg_q = 0
        self.max_q = 0
        self.std_q = 0

        self.max_ep_len = 0
        self.avg_ep_len = 0
        self.std_ep_len = 0

        self.eval_reward = -21.

    def update_averages(self,
                        rewards,
                        max_q_values,
                        q_values,
                        scores_eval,
                        episode_lengths,
                        max_episode_length):
        """
        Update the averages

        Args:
            rewards: deque
            max_q_values: deque
            q_values: deque
            scores_eval: list
        """
        if len(rewards) > 0 and not self.config.batch:
            self.avg_reward = np.mean(rewards)
            self.max_reward = np.max(rewards)
            self.std_reward = np.sqrt(np.var(rewards) / len(rewards))

            self.max_q      = np.mean(max_q_values)
            self.avg_q      = np.mean(q_values)
            self.std_q      = np.sqrt(np.var(q_values) / len(q_values))

            self.max_ep_len = max_episode_length
            self.avg_ep_len = np.mean(episode_lengths)
            self.std_ep_len = np.std(episode_lengths)

        if len(scores_eval) > 0:
            self.eval_reward = scores_eval[-1]

    def interact(self, replay_buffer, state, get_action):
        # replay memory stuff
        idx     = replay_buffer.store_frame(state)
        q_input = replay_buffer.encode_recent_observation()

        # chose action according to current Q and exploration
        best_action, q_values = self.get_best_action(q_input)

        # default for non-counting DQN
        state_action_counts = None
        hash_id = None

        # if this is a counting DQN,
        # get the hashed state and state action counts
        if hasattr(self, 'get_counts'):
            if not hasattr(self, 'A'):
                # initialize the hash
                k = self.config.sim_hash_k
                state_size = len(state.ravel())
                self.logger.info(
                    'Initializing SimHash w/ {} bits and state size {}'.format(
                        k, state_size
                    )
                )
                self.init_hash(
                    k=k, state_size=state_size, n_actions=self.env.action_space.n
                )

            # get estimate of how many times each action has been taken
            # from this state
            state_action_counts, hash_id = self.get_counts(state)

        action = get_action(
            best_action,
            q_values=q_values,
            state_action_counts=state_action_counts
        )

        # if this is a counting DQN,
        # store the hashed state and action
        if hash_id is not None:
            self.update_counts(hash_id, action)

        # perform action in env
        new_state, reward, done, info = self.env.step(action)

        # store the transition
        replay_buffer.store_effect(idx, action, reward, done)
        state = new_state

        return state, reward, done, q_values

    def train(self, exp_schedule, lr_schedule):
        """
        Performs training of Q

        Args:
            exp_schedule: Exploration instance s.t.
                exp_schedule.get_action(best_action) returns an action
            lr_schedule: Schedule for learning rate
        """

        # initialize replay buffer and variables
        if not self.config.batch:
            replay_buffer = ReplayBuffer(
                self.config.buffer_size, self.config.state_history
            )
        else:
            self.logger.info(
                'Loading replay buffer from {}'.format(self.config.buffer_path)
            )
            replay_buffer = ReplayBuffer.load(self.config.buffer_path)
            self.logger.info(
                'Loaded buffer with {} observations and {} in buffer'.format(
                    len(replay_buffer.obs), replay_buffer.num_in_buffer
                )
            )

        rewards = deque(maxlen=self.config.num_episodes_test)
        max_q_values = deque(maxlen=1000)
        q_values = deque(maxlen=1000)
        episode_lengths = deque(maxlen=1000)
        max_episode_length = 0
        self.init_averages()

        t = last_eval = last_record = 0  # time control of nb of steps
        scores_eval = []  # list of scores computed at iteration time
        scores_eval += [self.evaluate()]

        prog = Progbar(target=self.config.nsteps_train)

        # interact with environment
        while t < self.config.nsteps_train:
            total_reward = 0

            if not self.config.batch:
                state = self.env.reset()

            episode_step = 0
            avg_episode_length = (
                np.nan if len(episode_lengths) == 0 else np.mean(episode_lengths)
            )

            while True:
                t += 1
                episode_step += 1
                last_eval += 1
                last_record += 1
                if self.config.render_train:
                    self.env.render()

                if not self.config.batch:
                    get_action = functools.partial(
                        exp_schedule.get_action,
                        episode_num=len(episode_lengths),
                        episode_step=episode_step,
                        avg_episode_length=avg_episode_length
                    )
                    state, reward, done, _q_values = self.interact(
                        replay_buffer, state, get_action
                    )
                else:
                    reward = 0
                    done = True
                    _q_values = [0]

                # store q values
                max_q_values.append(max(_q_values))
                q_values.extend(list(_q_values))

                # perform a training step
                loss_eval, grad_eval = self.train_step(
                    t, replay_buffer, lr_schedule.epsilon
                )

                # logging stuff
                learning = (t > self.config.learning_start)
                learning_and_loggging = (
                    learning and
                    (t % self.config.log_freq == 0) and
                    (t % self.config.learning_freq == 0)
                )
                if learning_and_loggging:
                    self.update_averages(
                        rewards, max_q_values, q_values,
                        scores_eval, episode_lengths, max_episode_length
                    )
                    exp_schedule.update(t)
                    lr_schedule.update(t)
                    if len(rewards) > 0:
                        if self.config.batch:
                            exact = [
                                ("Loss", loss_eval),
                                ("Grads", grad_eval),
                                ("lr", lr_schedule.epsilon),
                            ]
                        else:
                            exact = [
                                ("Loss", loss_eval),
                                ("Avg_R", self.avg_reward),
                                ("Max_R", np.max(rewards)),
                                ("eps", exp_schedule.epsilon),
                                ("Grads", grad_eval),
                                ("Max_Q", self.max_q),
                                ("lr", lr_schedule.epsilon),
                                ("avg_ep_len", avg_episode_length)
                            ]

                        prog.update(t + 1, exact=exact)

                elif not learning and (t % self.config.log_freq == 0):
                    sys.stdout.write(
                        "\rPopulating the memory {}/{}...".format(
                            t, self.config.learning_start
                        )
                    )
                    sys.stdout.flush()

                # count reward
                total_reward += reward
                if done or t >= self.config.nsteps_train:
                    episode_lengths.append(episode_step)
                    if episode_step > max_episode_length:
                        max_episode_length = episode_step

                        # retrain the clusters every time the max episode
                        # length changes
                        if hasattr(self, 'reset_counts'):
                            self.reset_counts(
                                n_clusters=max_episode_length,
                                states=replay_buffer.get_encoded_states(),
                                actions=replay_buffer.get_actions()
                            )

                    break

            # updates to perform at the end of an episode
            rewards.append(total_reward)

            should_evaluate = (
                (t > self.config.learning_start) and
                (last_eval > self.config.eval_freq)
            )
            if should_evaluate:
                # evaluate our policy
                last_eval = 0
                print("")
                scores_eval.append(self.evaluate())

            should_record = (
                (t > self.config.learning_start) and
                self.config.record and
                (last_record > self.config.record_freq)
            )
            if should_record:
                self.logger.info("Recording...")
                last_record = 0
                self.record()

        # last words
        self.logger.info("- Training done.")
        self.save()
        scores_eval.append(self.evaluate())
        export_plot(scores_eval, "Scores", self.config.plot_output)

        if not self.config.batch:
            # save replay buffer
            self.logger.info(
                'Saving buffer to {}'.format(self.config.buffer_path)
            )
            replay_buffer.save(self.config.buffer_path)

    def train_step(self, t, replay_buffer, lr):
        """
        Perform training step

        Args:
            t: (int) nths step
            replay_buffer: buffer for sampling
            lr: (float) learning rate
        """
        loss_eval, grad_eval = 0, 0

        # perform training step
        should_train = (
            t > self.config.learning_start and
            t % self.config.learning_freq == 0
        )
        if should_train:
            loss_eval, grad_eval = self.update_step(t, replay_buffer, lr)

        # occasionaly update target network with q network
        if t % self.config.target_update_freq == 0:
            self.update_target_params()

        # occasionaly save the weights
        if (t % self.config.saving_freq == 0):
            self.save()

        return loss_eval, grad_eval

    def evaluate(self, env=None, num_episodes=None):
        """
        Evaluation with same procedure as the training
        """
        # log our activity only if default call
        if num_episodes is None:
            self.logger.info("Evaluating...")

        # arguments defaults
        if num_episodes is None:
            num_episodes = self.config.num_episodes_test

        if env is None:
            env = self.env

        # replay memory to play
        replay_buffer = ReplayBuffer(
            # self.config.buffer_size,
            num_episodes * 1000,
            self.config.state_history
        )
        rewards = []

        for i in range(num_episodes):
            total_reward = 0
            state = env.reset()
            while True:
                if self.config.render_test:
                    env.render()

                # store last state in buffer
                idx     = replay_buffer.store_frame(state)
                q_input = replay_buffer.encode_recent_observation()

                action = self.get_action(q_input)

                # perform action in env
                new_state, reward, done, info = env.step(action)

                # store in replay memory
                replay_buffer.store_effect(idx, action, reward, done)
                state = new_state

                # count reward
                total_reward += reward
                if done:
                    break

            # updates to perform at the end of an episode
            rewards.append(total_reward)

        avg_reward = np.mean(rewards)
        sigma_reward = np.sqrt(np.var(rewards) / len(rewards))

        if num_episodes > 1:
            msg = "Average reward: {:04.2f} +/- {:04.2f}".format(
                avg_reward, sigma_reward
            )
            self.logger.info(msg)

        return avg_reward

    def record(self):
        """
        Re create an env and record a video for one episode
        """
        env = gym.make(self.config.env_name)
        env = gym.wrappers.Monitor(env, self.config.record_path, video_callable=lambda x: True, resume=True)
        env = MaxAndSkipEnv(env, skip=self.config.skip_frame)
        env = PreproWrapper(
            env, prepro=greyscale, shape=(80, 80, 1), overwrite_render=self.config.overwrite_render
        )
        self.evaluate(env, 1)

    def run(self, exp_schedule, lr_schedule):
        """
        Apply procedures of training for a QN

        Args:
            exp_schedule: exploration strategy for epsilon
            lr_schedule: schedule for learning rate
        """
        # initialize
        self.initialize()

        # record one game at the beginning
        # if self.config.record:
        #     self.record()

        # model
        self.train(exp_schedule, lr_schedule)

        # record one game at the end
        if self.config.record:
            self.record()
