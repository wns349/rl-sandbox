import gym
import random
import numpy as np
import os
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential


class DQNAgent(object):
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.discount_factor = 0.99
        self.learning_rate = 1e-1
        self.epsilon = 0.1
        # self.epsilon_decay = 0.999
        self.epsilon_decay = 1
        self.epsilon_min = 0.01
        self.train_start = 100
        self.batch_size = 64

        self.memory = deque(maxlen=2000)

        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(24,
                        input_dim=self.state_size,
                        activation="relu",
                        kernel_initializer="he_uniform"))
        model.add(Dense(24,
                        activation="relu",
                        kernel_initializer="he_uniform"))
        model.add(Dense(self.action_size,
                        activation="linear",
                        kernel_initializer="he_uniform"))
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def load_model(self, path_to_model):
        if os.path.exists(path_to_model):
            self.model.load_weights(path_to_model)
            self.update_target_model()

    def save_model(self, path_to_model):
        self.model.save_weights(path_to_model)

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:  # exploration
            return random.randrange(self.action_size)
        else:
            q_values = self.model.predict(state)
            return np.argmax(q_values[0])  # TODO: what if there are multiple indices?

    def add_sample(self, state, action, reward, next_state, is_done):
        self.memory.append((state, action, reward, next_state, is_done))
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def train_model(self):
        if len(self.memory) < self.train_start:
            return  # not enough experiences yet
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        _state = np.zeros((batch_size, self.state_size))
        _target_state = np.zeros((batch_size, self.state_size))

        # make input for dnn
        for i, experience in enumerate(mini_batch):
            s, a, r, sp, t = experience  # (state, action, reward, next_state, is_done)
            _state[i] = s
            _target_state[i] = sp

        _pred = self.model.predict(_state)
        _target_pred = self.target_model.predict(_target_state)

        for i, experience in enumerate(mini_batch):
            s, a, r, sp, t = experience  # (state, action, reward, next_state, is_done)
            if t:
                _pred[i][a] = r
            else:
                _pred[i][a] = r + self.discount_factor * (np.max(_target_pred[i]))

        self.model.fit(_state, _pred, batch_size=batch_size, epochs=1, verbose=0)


def reshape_state(state):
    return np.expand_dims(state, axis=0)


def run_cartpole(total_episodes=1000, save_weights_interval=50, weights_path="./cartpole.h5"):
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)
    agent.load_model(weights_path)
    scores = []

    for episode in range(total_episodes):
        done = False
        state = env.reset()
        state = reshape_state(state)
        score = 0
        while not done:
            env.render()
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = reshape_state(next_state)

            agent.add_sample(state, action, reward, next_state, done)
            agent.train_model()

            score += reward
            state = next_state

        # episode is done
        agent.update_target_model()
        scores.append(score)

        print("episode: {} / score: {} / memory length: {} / epsilon: {}".format(episode, score, len(agent.memory),
                                                                                 agent.epsilon))

        if episode > 0 and episode % save_weights_interval == 0:
            agent.save_model(weights_path)

    env.close()


if __name__ == "__main__":
    run_cartpole()
