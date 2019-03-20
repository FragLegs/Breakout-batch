import tensorflow as tf
from core.linear import Linear


class NatureQN(Linear):
    """
    Implementing DeepMind's Nature paper. Here are the relevant urls.
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
    """

    def get_q_values_op(self, state, scope, reuse=False):
        """
        Returns Q values for all actions

        Args:
            state: (tf tensor)
                shape = (batch_size, img height, img width, nchannels)
            scope: (string) scope name, that specifies if target network or not
            reuse: (bool) reuse of variables in the scope

        Returns:
            out: (tf tensor) of shape = (batch_size, num_actions)
        """
        # this information might be useful
        num_actions = self.env.action_space.n

        # From the Model Architecture section of the Nature paper:
        #
        # The exact architecture, shown schematically in Fig. 1, is as follows.
        #
        # The input to the neural network consists of an 84 x 84 x 4 (NOTE: we use 80x80x4)
        # image produced by the preprocessing map w.
        #
        # The first hidden layer convolves 32 filters of 8 x 8
        # with stride 4 with the input image
        # and applies a rectifier nonlinearity.
        #
        # The second hidden layer convolves 64 filters of 4 x 4
        # with stride 2,
        # again followed by a rectifier nonlinearity.
        #
        # This is followed by a third convolutional layer
        # that convolves 64 filters of 3 x 3
        # with stride 1
        # followed by a rectifier.
        #
        # The final hidden layer is fully-connected
        # and consists of 512 rectifier units.
        #
        # The output layer is a fully-connected linear layer
        # with a single output for each valid action.
        with tf.variable_scope(scope, reuse=reuse):
            # The first hidden layer convolves 32 filters of 8 x 8
            # with stride 4 with the input image
            # and applies a rectifier nonlinearity.
            layer1 = tf.nn.relu(tf.layers.conv2d(
                inputs=state,
                filters=32,
                kernel_size=(8, 8),
                strides=4,
                padding='same',
                # data_format='channels_last',
                # dilation_rate=(1, 1),
                # activation=None,
                # use_bias=True,
                # kernel_initializer=None,
                # bias_initializer=tf.zeros_initializer(),
                # kernel_regularizer=None,
                # bias_regularizer=None,
                # activity_regularizer=None,
                # kernel_constraint=None,
                # bias_constraint=None,
                # trainable=True,
                name='conv_layer1',
                # reuse=None
            ))

            # The second hidden layer convolves 64 filters of 4 x 4
            # with stride 2,
            # again followed by a rectifier nonlinearity.
            layer2 = tf.nn.relu(tf.layers.conv2d(
                inputs=layer1,
                filters=64,
                kernel_size=(4, 4),
                strides=2,
                padding='same',
                # data_format='channels_last',
                # dilation_rate=(1, 1),
                # activation=None,
                # use_bias=True,
                # kernel_initializer=None,
                # bias_initializer=tf.zeros_initializer(),
                # kernel_regularizer=None,
                # bias_regularizer=None,
                # activity_regularizer=None,
                # kernel_constraint=None,
                # bias_constraint=None,
                # trainable=True,
                name='conv_layer2',
                # reuse=None
            ))

            # This is followed by a third convolutional layer
            # that convolves 64 filters of 3 x 3
            # with stride 1
            # followed by a rectifier.
            layer3 = tf.nn.relu(tf.layers.conv2d(
                inputs=layer2,
                filters=64,
                kernel_size=(3, 3),
                strides=1,
                padding='same',
                # data_format='channels_last',
                # dilation_rate=(1, 1),
                # activation=None,
                # use_bias=True,
                # kernel_initializer=None,
                # bias_initializer=tf.zeros_initializer(),
                # kernel_regularizer=None,
                # bias_regularizer=None,
                # activity_regularizer=None,
                # kernel_constraint=None,
                # bias_constraint=None,
                # trainable=True,
                name='conv_layer3',
                # reuse=None
            ))

            # The final hidden layer is fully-connected
            # and consists of 512 rectifier units.
            layer4 = tf.nn.relu(tf.layers.dense(
                inputs=tf.layers.flatten(layer3),
                units=512,
                # activation=None,
                # use_bias=True,
                # kernel_initializer=None,
                # bias_initializer=tf.zeros_initializer(),
                # kernel_regularizer=None,
                # bias_regularizer=None,
                # activity_regularizer=None,
                # kernel_constraint=None,
                # bias_constraint=None,
                # trainable=True,
                name='dense_layer4',
                # reuse=None
            ))

            # if scope == 'q':
            #     self.encoded_state = layer4

            # The output layer is a fully-connected linear layer
            # with a single output for each valid action.
            out = tf.layers.dense(
                inputs=layer4,
                units=num_actions,
                # activation=None,
                # use_bias=True,
                # kernel_initializer=None,
                # bias_initializer=tf.zeros_initializer(),
                # kernel_regularizer=None,
                # bias_regularizer=None,
                # activity_regularizer=None,
                # kernel_constraint=None,
                # bias_constraint=None,
                # trainable=True,
                name='dense_layer5',
                # reuse=None
            )
        return out
