# -*- coding: utf-8 -*-
import inspect
import logging

import numpy as np

import schedule

log = logging.getLogger(__name__)


class LinearRandom(schedule.LinearSchedule):
    def __init__(self, env, eps_begin, eps_end, nsteps):
        """
        Args:
            env: gym environment
            eps_begin: float
                initial exploration rate
            eps_end: float
                final exploration rate
            nsteps: int
                number of steps taken to linearly decay eps_begin to eps_end
        """
        self.env = env
        super(LinearRandom, self).__init__(eps_begin, eps_end, nsteps)

    def get_action(self, best_action, **kwargs):
        """
        Returns a random action with prob epsilon, otherwise returns the best_action

        Args:
            best_action: int
                best action according some policy
        Returns:
            an action
        """
        return (
            best_action if self.do_greedy(**kwargs)  # p(1 - epsilon)
            else self.random_action(**kwargs)     # p(epsilon)
        )

    def do_greedy(self, **kwargs):
        return np.random.rand() > self.epsilon

    def random_action(self, **kwargs):
        return self.env.action_space.sample()


class LinearInverseCount(LinearRandom):
    def random_action(self, state_action_counts, **kwargs):
        inverse_square_counts = 1.0 / np.sqrt(state_action_counts)
        probs = inverse_square_counts / inverse_square_counts.sum()
        actions = range(len(state_action_counts))
        return np.random.choice(actions, p=probs)


class ShapedRandom(LinearRandom):
    def do_greedy(self, episode_num, episode_step, avg_episode_length, **kwargs):
        """
        Determines whether to return the greedy action or a random action

        Args:
            best_action: int
                best action according some policy
        Returns:
            bool: True == do greedy
        """
        if np.isnan(avg_episode_length):
            return super(ShapedRandom, self).do_greedy()

        if episode_num % 2 == 0:
            eps_modifier = float(episode_step) / avg_episode_length
        else:
            eps_modifier = (float(episode_step) - avg_episode_length) / avg_episode_length

        prob_explore = max(0.0, min(1.0, eps_modifier * self.epsilon))

        return np.random.rand() > prob_explore


class ShapedInverseCount(ShapedRandom, LinearInverseCount):
    pass


class NoExploration(LinearRandom):
    def do_greedy(self, **kwargs):
        return True


class StochasticRandom(LinearRandom):
    def do_greedy(self, **kwargs):
        return False

    def random_action(self, q_values, **kwargs):
        e_q = np.exp(q_values - np.max(q_values))
        softmax = e_q / e_q.sum()
        actions = range(len(q_values))
        return np.random.choice(actions, p=softmax)


class StochasticInverseCount(LinearRandom):
    def do_greedy(self, **kwargs):
        return False

    def random_action(self, q_values, state_action_counts, beta, **kwargs):
        e_q = np.exp(q_values - np.max(q_values))
        softmax = e_q / e_q.sum()

        inverse_square_counts = 1.0 / np.sqrt(state_action_counts)
        ic_probs = inverse_square_counts / inverse_square_counts.sum()

        probs = softmax + (beta * ic_probs)
        probs = probs / probs.sum()

        actions = range(len(q_values))
        return np.random.choice(actions, p=probs)


EXPLORATION_MAP = {
    key: value for key, value in globals().items()
    if inspect.isclass(value) and issubclass(value, schedule.LinearSchedule)
}
