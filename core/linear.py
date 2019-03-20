import tensorflow as tf
from core.deep_q_learning import DQN


class Linear(DQN):
    """
    Implement Fully Connected with Tensorflow
    """

    def add_placeholders_op(self):
        """
        Adds placeholders to the graph

        These placeholders are used as inputs to the rest of the model and will be fed
        data during training.
        """
        # this information might be useful
        state_shape = list(self.env.observation_space.shape)
        height, width, n_channels = state_shape
        n_history = self.config.state_history

        self.s = tf.placeholder(
            tf.uint8,
            shape=(None, height, width, n_channels * n_history),
            name='states'
        )

        self.a = tf.placeholder(
            tf.int32,
            shape=(None,),
            name='actions'
        )

        self.r = tf.placeholder(
            tf.float32,
            shape=(None,),
            name='rewards'
        )

        self.sp = tf.placeholder(
            tf.uint8,
            shape=(None, height, width, n_channels * n_history),
            name='states_prime'
        )

        self.done_mask = tf.placeholder(
            tf.bool,
            shape=(None,),
            name='done_mask'
        )

        self.lr = tf.placeholder(
            tf.float32,
            shape=(),
            name='learning_rate'
        )

    def get_q_values_op(self, state, scope, reuse=False):
        """
        Returns Q values for all actions

        Args:
            state: (tf tensor)
                shape = (batch_size, img height, img width, nchannels x config.state_history)
            scope: (string) scope name, that specifies if target network or not
            reuse: (bool) reuse of variables in the scope

        Returns:
            out: (tf tensor) of shape = (batch_size, num_actions)
        """
        # this information might be useful
        num_actions = self.env.action_space.n

        with tf.variable_scope(scope, reuse=reuse):
            out = tf.layers.dense(
                inputs=tf.layers.flatten(state),
                units=num_actions,
                name='fully_connected'
            )

        return out

    def add_update_target_op(self, q_scope, target_q_scope):
        """
        update_target_op will be called periodically
        to copy Q network weights to target Q network

        Remember that in DQN, we maintain two identical Q networks with
        2 different sets of weights. In tensorflow, we distinguish them
        with two different scopes. If you're not familiar with the scope mechanism
        in tensorflow, read the docs
        https://www.tensorflow.org/programmers_guide/variable_scope

        Periodically, we need to update all the weights of the Q network
        and assign them with the values from the regular network.
        Args:
            q_scope: (string) name of the scope of variables for q
            target_q_scope: (string) name of the scope of variables
                        for the target network
        """
        q = tf.trainable_variables(scope=q_scope)
        target_q = tf.trainable_variables(scope=target_q_scope)
        self.update_target_op = tf.group(*[
            tf.assign(ref=_t, value=_q, validate_shape=True)
            for _t, _q in zip(target_q, q)
        ])

    def add_loss_op(self, q, target_q):
        """
        Sets the loss of a batch, self.loss is a scalar

        Args:
            q: (tf tensor) shape = (batch_size, num_actions)
            target_q: (tf tensor) shape = (batch_size, num_actions)
        """
        # you may need this variable
        num_actions = self.env.action_space.n

        # q_samp = r if done
        #        = r + gamma * max_a' target_q
        r = self.r
        gamma = tf.constant(self.config.gamma)
        target_q = self.target_q
        q = self.q

        # use not_done to select r or r + gamma * max_a' target_q
        # for all samples in the batch
        not_done = tf.cast(tf.logical_not(self.done_mask), dtype=tf.float32)

        q_samp = r + (not_done * (gamma * tf.reduce_max(target_q, axis=1)))

        # get q(s,a) by one-hot encoding the selected action for all
        # samples in the batch
        batch_actions = tf.one_hot(indices=self.a, depth=num_actions)
        estimate = tf.reduce_sum(q * batch_actions, axis=1)

        # L = avg((q_samp - estimate)^2)
        self.loss = tf.reduce_mean(tf.squared_difference(q_samp, estimate))

    def add_optimizer_op(self, scope):
        """
        Set self.train_op and self.grad_norm
        Args:
            scope: (string) scope name, that specifies if target network or not
        """
        adam = tf.train.AdamOptimizer(
            learning_rate=self.lr,
            # beta1=0.9,
            # beta2=0.999,
            # epsilon=1e-08,
            # use_locking=False,
            # name='Adam'
        )

        # a list of (gradient, variable) pairs
        grads_and_vars = adam.compute_gradients(
            loss=self.loss,
            var_list=tf.trainable_variables(scope=scope),
            # gate_gradients=GATE_OP,
            # aggregation_method=None,
            # colocate_gradients_with_ops=False,
            # grad_loss=None,
            # stop_gradients=None,
            # scale_loss_by_num_towers=None
        )

        # clip gradients if desired
        if self.config.grad_clip:
            grads_and_vars = [
                (tf.clip_by_norm(g, self.config.clip_val), v)
                for g, v in grads_and_vars
            ]

        # apply gradients
        self.train_op = adam.apply_gradients(
            grads_and_vars,
            # global_step=None,
            name='apply_gradients'
        )

        # compute global norm (potentially after clipping)
        self.grad_norm = tf.global_norm([g for g, v in grads_and_vars])
