import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from numpy import float32, int32, zeros, bool, random, array


class DuelingDeepQNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims, fc2_dims):
        super(DuelingDeepQNetwork, self).__init__()

        self.dense1 = keras.layers.Dense(fc1_dims, activation='relu')
        self.dense2 = keras.layers.Dense(fc2_dims, activation='relu')
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

    def save(self, path):
        self.save_weights(path)

    def load(self, path):
        self.load_weights(path)


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
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones


class Agent():
    def __init__(self, lr, gamma, n_actions, epsilon, batch_size, input_dims, epsilon_dec=1e-3,
                 eps_end=0.01, mem_size=1000000, fc1_dims=256, fc2_dims=256, replace_target_cnt=500):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.eps_end = eps_end
        self.batch_size = batch_size
        self.replace = replace_target_cnt
        self.learn_step_counter = 0
        self.memory = ReplayBuffer(mem_size, input_dims)
        self.q_val = DuelingDeepQNetwork(n_actions, fc1_dims, fc2_dims)
        self.q_next = DuelingDeepQNetwork(n_actions, fc1_dims, fc2_dims)

        self.q_val.compile(optimizer=Adam(lr=lr), loss='mse')
        self.q_next.compile(optimizer=Adam(lr=lr), loss='mse')

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def choose_action(self, observation):
        if random.random() > self.epsilon:
            state = array([observation])
            actions = self.q_val.advantage(state)
            action = tf.argmax(actions, axis=1).numpy()
            action = action[0]
        else:
            action = random.choice(self.action_space)
        return action

    def save_model(self, path):
        self.q_val.save(path+'_q_val')
        self.q_next.save(path+'_q_next')

    def load_model(self, path):
        self.q_val.load(path+'_q_val')
        self.q_next.load(path+'_q_next')

    def learn(self):
        if self.memory.men_cntr < self.batch_size:
            return

        if self.learn_step_counter % self.replace == 0:
            self.q_next.set_weights(self.q_val.get_weights())

        state, action, reward, state_, done = self.memory.sample_buffer(
            self.batch_size)

        q_pred = self.q_val(state)
        q_next = self.q_next(state_)
        q_target = q_pred.numpy()
        max_actions = tf.argmax(q_next, axis=1)

        for idx, terminal in enumerate(done):

            q_target[idx, action[idx]] = reward[idx] + \
                self.gamma * q_target[idx, max_actions[idx]]*(1-int(done[idx]))

        self.q_val.train_on_batch(state, q_target)
        self.epsilon = self.epsilon - \
            self.epsilon_dec if self.epsilon > self.eps_end else self.eps_end
        self.learn_step_counter += 1
