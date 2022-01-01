import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from numpy import float32, int32, zeros, bool, random


class DuelingDeepQNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims, fc2_dims):
        super(DuelingDeepQNetwork, self).__init__()

        self.dense1 = keras.layers.Dense(fc1_dims, activation='relu')
        self.dense1 = keras.layers.Dense(fc2_dims, activation='relu')
        self.V = keras.layers.Dense(1, activation=None)
        self.A = keras.layers.Dense(n_actions, activation=None)

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        V = self.V(x)
        A = self.A(x)
        Q = (V + (A - tf.math.reduce_mean(A, axis=1, keepdims=True)))
        return Q

    def advantage(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        A = self.A(x)
        return A


class ReplayBuffer():
    def __init__(self, max_size, input_shape):
        self.men_size = max_size
        self.men_cntr = 0

        self.state_memory = zeros(
            (self.men_size, *input_shape), dtype=float32)
        self.new_state_memory = zeros(
            (self.men_size, *input_shape), dtype=float32)

        self.action_memory = zeros(self.men_size, dtype=int32)
        self.reward_memory = zeros(self.men_size, dtype=float32)
        self.terminal_memory = zeros(self.men_size, dtype=bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.men_cntr % self.men_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.men_cntr += 1

    def sample_buffer(self, batch_size):
        max_men = min(self.men_cntr, self.men_size)
        batch = random.choice(max_men, batch_size, replace=False)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.
