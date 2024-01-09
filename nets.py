import tensorflow as tf
from tensorflow.keras.layers import LSTMCell
import numpy as np


def build_feature_extractor(input_, h, c):
    # scale inputs from 0-255 to 0-1
    input_ = tf.to_float(input_)

    # CNN layers
    conv1 = tf.contrib.layers.conv2d(
        input_,
        16,  # output features maps
        8,  # kernel size
        4,  # stride
        activation_fn=tf.nn.relu,
        scope="conv1")

    conv2 = tf.contrib.layers.conv2d(
        conv1,
        32,  # output features maps
        4,  # kernel size
        2,  # stride
        activation_fn=tf.nn.relu,
        scope="conv2"
    )

    # image to feature vector
    flat = tf.contrib.layers.flatten(conv2)

    # dense layer (fully connected)
    fc1 = tf.contrib.layers.fully_connected(
        inputs=flat,
        num_outputs=256,
        scope="fc1")
    lstm = LSTMCell(256)
    rnn_out, (ht, ct) = lstm(fc1, states=[h, c])

    return rnn_out, ht, ct


class Network:
    def __init__(self, num_outputs, reg=0.01):
        self.num_outputs = num_outputs

        # Graph inputs
        # after resizing we have 4 consecutive frames of 84x84 size
        self.states = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name="X")
        # Advantage = G - V(s)
        self.advantage = tf.placeholder(shape=[None], dtype=tf.float32, name="Y")
        # selected actions
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")
        self.targets = tf.placeholder(shape=[None], dtype=tf.float32, name="Y")

        self.h = tf.zeros((1, 256))
        self.c = tf.zeros((1, 256))

        rnn_out, self.h, self.c = build_feature_extractor(self.states, self.h, self.c)




        with tf.variable_scope("policy"):
            self.logits = tf.contrib.layers.fully_connected(rnn_out, num_outputs, activation_fn=None)
            self.probs = tf.nn.softmax(self.logits)

            # Sample an action
            cdist = tf.distributions.Categorical(logits=self.logits)
            self.sample_action = cdist.sample()
            # Add regularization to increase exploration
            self.entropy = -tf.reduce_sum(self.probs * tf.log(tf.maximum(self.probs, 1e-20)), axis=1)

            # Get the predictions for the chosen actions only
            batch_size = tf.shape(self.states)[0]
            gather_indices = tf.range(batch_size) * tf.shape(self.probs)[1] + self.actions
            self.selected_action_probs = tf.gather(tf.reshape(self.probs, [-1]), gather_indices)

            self.loss = tf.log(tf.maximum(self.selected_action_probs, 1e-20)) * self.advantage + reg * self.entropy
            self.loss = -tf.reduce_sum(self.loss, name="loss")
            self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
            # training

            # we'll need these later for running gradient descent steps
            self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
            self.grads_and_vars = [[grad, var] for grad, var in self.grads_and_vars if grad is not None]
        with tf.variable_scope("value"):
            self.vhat = tf.contrib.layers.fully_connected(
                inputs=rnn_out,
                num_outputs=1,
                activation_fn=None
            )
            self.vhat = tf.squeeze(self.vhat, squeeze_dims=[1], name="vhat")
            self.v_loss = tf.squared_difference(self.vhat, self.targets)
            self.v_loss = tf.reduce_sum(self.v_loss, name="loss")
            self.voptimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)

            # we'll need these later for running gradient descent steps

            self.v_grads_and_vars = self.voptimizer.compute_gradients(self.v_loss)
            self.v_grads_and_vars = [[grad, var] for grad, var in self.v_grads_and_vars if grad is not None]
            # Sample an action


# Use this to create networks, to ensure they are created in the correct order
def create_networks(num_outputs):
    network = Network(num_outputs=num_outputs)

    return network







