import gym
import random
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential


class DQNAgent(object):
    def __init__(self, state_size, action_size, is_train):
        self.state_size = state_size
        self.action_size = action_size
        self.is_train = is_train

        self.discount_factor = 0.9
        self.learning_rate = 1e-2
        self.learning_rate_decay = 1e-2
        self.epsilon = 0.1
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.001
        self.train_start = 100
        self.batch_size = 64

        self.memory = deque(maxlen=10000)

        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(16,
                        input_dim=self.state_size,
                        activation="relu"))
        model.add(Dense(32,
                        activation="relu"))
        model.add(Dense(self.action_size,
                        activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate, decay=self.learning_rate_decay))
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
        if self.is_train and np.random.rand() <= self.epsilon:  # exploration
            return random.randrange(self.action_size)
        else:
            q_values = self.model.predict(state)
            best_qs = np.argwhere(q_values == np.amax(q_values[0])).flatten()
            return np.random.choice(best_qs)

    def add_sample(self, state, action, reward, next_state, is_done):
        self.memory.append((state, action, reward, next_state, is_done))

    def train_model(self):
        if len(self.memory) < self.train_start:
            return  # not enough experiences yet
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        states = []
        next_states = []
        # make input for dnn
        for s, a, r, n, t in mini_batch:
            states.append(s[0])
            next_states.append(n[0])

        states = np.array(states)
        next_states = np.array(next_states)

        target = self.model.predict(states)
        qhat = self.target_model.predict(next_states)

        for i, experience in enumerate(mini_batch):
            s, a, r, n, t = experience  # (state, action, reward, next_state, is_done)
            if t:
                target[i][a] = r
            else:
                target[i][a] = r + self.discount_factor * (np.max(qhat[i]))

        self.model.fit(states, target, batch_size=batch_size, epochs=1, verbose=0)

    def update_epsilon(self, episode, step):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)


def reshape_state(state):
    return np.expand_dims(state, axis=0)


def plot_scores(scores):
    plt.plot(scores)
    plt.ylabel("Scores")
    plt.xlabel("Episodes")
    plt.plot()
    plt.savefig("./cartpole.png")
    plt.show()


def run_cartpole(total_episodes=1000,
                 save_weights_interval=50,
                 weights_path="./cartpole.h5",
                 render=True,
                 train=True,
                 train_step_interval=20,
                 target_update_episode_interval=20):
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size, train)
    agent.load_model(weights_path)
    scores = []
    mva_score = 0
    for episode in range(total_episodes):
        done = False
        state = env.reset()
        state = reshape_state(state)
        score = 0
        step = 0
        while not done:
            if render:
                env.render()

            step += 1
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = reshape_state(next_state)

            agent.add_sample(state, action, reward, next_state, done)
            score += reward
            state = next_state

            if train and step % train_step_interval == 0:
                agent.train_model()

        # episode is done
        if train:
            if episode > 0 and episode % target_update_episode_interval == 0:
                agent.update_target_model()
            agent.update_epsilon(episode, step)
            scores.append(score)
            mva_score = mva_score * 0.9 + score * 0.1 if len(scores) > 5 else np.average(scores)
        print("Game over: {} / step: {} / score: {} / average_score: {:.5f} / epsilon: {:.5f}".format(episode,
                                                                                                      step,
                                                                                                      score,
                                                                                                      mva_score,
                                                                                                      agent.epsilon))

        if episode > 0 and episode % save_weights_interval == 0:
            agent.save_model(weights_path)
    plot_scores(scores)

    env.close()


if __name__ == "__main__":
    run_cartpole(render=False,
                 train=True,
                 total_episodes=5000,
                 save_weights_interval=5000,
                 train_step_interval=1,
                 target_update_episode_interval=1)
