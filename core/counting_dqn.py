# -*- coding: utf-8 -*-
import collections
import logging

import numpy as np

from core.nature import NatureQN

log = logging.getLogger(__name__)


class CountingDQN(NatureQN):
    def init_hash(self, k, state_size, n_actions):
        """
        Initializes a simhash function for discrtizing states and counting
        actions

        Parameters
        ----------
        k : int
            The dimensionality of the hash
        state_size : int
            The total number of parameters in the state
        n_actions : int
            The number of possible actions
        """
        def init_counts():
            # Everything starts with count of 1 to avoid divide by zero
            return [1] * n_actions

        self.A = np.random.normal(size=(k, state_size))
        self.counts = collections.defaultdict(init_counts)

    def hash_state(self, state):
        """
        Hashes the state (80 x 80 x 4) pixels into 32 8-bit uints using
        SimHash

        Parameters
        ----------
        state : numpy array
            Observation from gym environment
        """
        sims = np.dot(self.A, state.ravel().reshape(-1, 1)).ravel()
        return tuple(np.packbits(sims >= 0))

    def get_counts(self, state):
        """
        Get an estimated count of the actions seen in this state

        Parameters
        ----------
        state : encoded state observation

        Returns
        -------
        list of int
            The estimated counts for each action in this state
        np.array of uint8
            hash_id (for efficiency)
        """
        # get hash
        hash_id = self.hash_state(state)
        return self.counts[hash_id], hash_id

    def update_counts(self, hash_id, action):
        """
        Updates the estimated counts for this state action pair

        Parameters
        ----------
        hash_id : np.array of uint8
            Hash of the state observation
        action : int
            A taken action
        """
        self.counts[hash_id][action] += 1
