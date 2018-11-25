import gym
import os
import cv2
import random
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, Flatten, Dense, InputLayer, Lambda
from keras.optimizers import RMSprop
from keras import backend as K
from keras.models import Sequential
from collections import deque


class AtariDDQNAgent(object):
    def __init__(self, state_shape, action_size, log_dir="./log", is_test=False):
        self.state_shape = state_shape
        self.action_size = action_size

        # hyper-parameters
        self.exploration_steps = 1e6
        self.epsilon = 1. if not is_test else 0.001
        self.epsilon_min = 0.1 if not is_test else 0.001
        self.epsilon_decay_step = (self.epsilon - self.epsilon_min) / (self.exploration_steps * 1.)
        self.memory = deque(maxlen=1000000)
        self.batch_size = 32
        self.train_start = 5000
        self.learning_rate = 0.00025
        self.discount_factor = 0.99
        self.is_test = is_test

        # build model
        self.model = self.build_model(state_shape, action_size)
        self.target_model = self.build_model(state_shape, action_size)
        self.optimizer = self.build_optimizer()

        # tf summary for tensorboard
        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)
        self.summary_placeholder, self.summary_update_ops, self.summary_op = self.setup_summary()
        self.summary_writer = tf.summary.FileWriter(log_dir, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def preprocess(self, img, new_shape=(84, 84)):
        new_img = cv2.resize(img, new_shape)
        new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2GRAY)
        new_img.astype(np.uint8)
        return new_img

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def load_weights(self, path_to_weights):
        if os.path.exists(path_to_weights):
            self.model.load_weights(path_to_weights)
            self.update_target_model()

    def save_weights(self, path_to_weights):
        self.model.save_weights(path_to_weights)

    def update_epsilon(self):
        self.epsilon = max(self.epsilon - self.epsilon_decay_step, self.epsilon_min)

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        else:
            q_values = self.model.predict(np.expand_dims(state, axis=0))
            return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, is_done):
        self.memory.append((state, action, reward, next_state, is_done))

    def train_model(self):
        if len(self.memory) < self.train_start or self.is_test:
            return  # not enough experiences yet
        self.update_epsilon()

        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        states = []
        next_states = []
        actions = []
        for s, a, r, n, t in mini_batch:  # state, action, reward, next state, terminated
            states.append(s)
            next_states.append(n)
            actions.append(a)

        states = np.array(states)
        next_states = np.array(next_states)
        actions = np.array(actions)
        targets = np.zeros((batch_size,))

        qs = self.model.predict(next_states)
        target_qs = self.target_model.predict(next_states)

        for i, b in enumerate(mini_batch):
            s, a, r, n, t = b
            if t:
                targets[i] = r
            else:
                targets[i] = r + self.discount_factor * target_qs[i][np.argmax(qs[i])]

        loss = self.optimizer([states, actions, targets])

    def build_model(self, input_shape, output_shape):
        model = Sequential()
        model.add(InputLayer(input_shape))
        model.add(Lambda(lambda x: x / 255.0))
        model.add(Conv2D(32, 8, strides=(4, 4), activation="relu"))
        model.add(Conv2D(64, 4, strides=(2, 2), activation="relu"))
        model.add(Conv2D(64, 3, strides=(1, 1), activation="relu"))
        model.add(Flatten())
        model.add(Dense(512, activation="relu"))
        model.add(Dense(output_shape, activation="linear"))
        model.summary()
        return model

    def build_optimizer(self):
        action = K.placeholder(shape=(None,), dtype="int32")
        y = K.placeholder(shape=(None,), dtype="float32")
        out = self.model.output

        # huber loss
        one_hot = K.one_hot(action, self.action_size)
        q_value = K.sum(out * one_hot, axis=1)
        error = y - q_value
        condition = K.abs(error) < 1.0
        squared_loss = 0.5 * K.square(error)
        linear_loss = K.abs(error) - 0.5
        clipped_error = tf.where(condition, squared_loss, linear_loss)
        loss = K.mean(clipped_error)

        optimizer = RMSprop(lr=self.learning_rate, epsilon=0.01)
        updates = optimizer.get_updates(self.model.trainable_weights, [], loss)
        train_op = K.function([self.model.input, action, y], [loss], updates=updates)
        return train_op

    def setup_summary(self):
        # for tensorboard
        # TODO
        episode_duration = tf.Variable(0.)
        episode_avg_max_q = tf.Variable(0.)

        tf.summary.scalar("Duration/Episode", episode_duration)
        tf.summary.scalar("Average Max Q/Episode", episode_avg_max_q)

        variables = [episode_duration, episode_avg_max_q]
        placeholders = [tf.placeholder(tf.float32) for _ in range(len(variables))]
        update_ops = [variables[i].assign(placeholders[i]) for i in range(len(variables))]
        summary_op = tf.summary.merge_all()
        return placeholders, update_ops, summary_op


def main(env_name="BreakoutDeterministic-v4",
         weights_path="ddqn.h5",
         episodes=10000,
         render=True,
         update_target_rate=10000,
         test=False,
         log_dir="./log",
         save_weights_episode=1000):
    env = gym.make(env_name)
    frame = env.reset()
    print(env.observation_space)
    print(env.action_space.n)

    agent = AtariDDQNAgent(state_shape=(84, 84, 4), action_size=env.action_space.n, is_test=test, log_dir=log_dir)
    agent.load_weights(weights_path)

    global_step = 0
    for episode in range(1, episodes + 1):
        env.reset()

        frame, _, _, info = env.step(1)

        # DeepMind's idea: do nothing for a while to avoid sub-optimal?
        for _ in range(random.randint(1, 30)):
            frame, _, _, info = env.step(env.action_space.sample())

        frame = agent.preprocess(frame)  # [h, w]
        state = np.stack((frame, frame, frame, frame), axis=-1)  # [h, w, 4]

        lives = info["ale.lives"]
        max_qs = 0
        step = 0
        score = 0
        done = False
        while not done:
            if render:
                env.render()
            global_step += 1
            step += 1

            action = agent.get_action(state)

            frame, reward, done, info = env.step(action)
            frame = agent.preprocess(frame)  # [h, w]
            frame = np.expand_dims(frame, axis=-1)  # [h, w, 1]
            next_state = np.append(frame, state[..., :3], axis=-1)  # [h, w, 4]

            dead = lives != info["ale.lives"]  # agent is dead, but episode is not over
            lives = info["ale.lives"]
            max_qs += np.amax(agent.model.predict(np.expand_dims(state, axis=0))[0])

            reward = np.clip(reward, -1., 1.)
            score += reward

            agent.remember(state, action, reward, next_state, dead)
            agent.train_model()
            if global_step % update_target_rate == 0:
                agent.update_target_model()

            state = next_state
        # done
        if global_step > agent.train_start:
            summary_vals = [step, max_qs / float(step)]
            for i in range(len(summary_vals)):
                agent.sess.run(agent.summary_update_ops[i], feed_dict={
                    agent.summary_placeholder[i]: float(summary_vals[i])
                })
            summary_str = agent.sess.run(agent.summary_op)
            agent.summary_writer.add_summary(summary_str, episode)

        print("episode: ", episode,
              " score: ", score,
              " avg max q: ", max_qs / float(step),
              " memory: ", len(agent.memory),
              " epsilon: ", agent.epsilon,
              " global step: ", global_step)

        if save_weights_episode > 0 and episode % save_weights_episode == 0:
            agent.save_weights(weights_path)

    env.close()
    print("Bye")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", dest="env_name", default="MsPacmanDeterministic-v4", type=str)
    parser.add_argument("--weights_path", dest="weights_path", default="pacman.h5", type=str)
    parser.add_argument("--render", dest="render", action="store_true", default=False)
    parser.add_argument("--episodes", dest="episodes", default=10000, type=int)
    parser.add_argument("--target_rate", dest="target_rate", default=5000, type=int)
    parser.add_argument("--test", dest="test", action="store_true", default=False)
    parser.add_argument("--log_dir", dest="log_dir", default="./log", type=str)
    parser.add_argument("--save_weights_episode", dest="save_weights_episode", default=1000, type=int)
    args = parser.parse_args()
    main(env_name=args.env_name,
         weights_path=args.weights_path,
         episodes=args.episodes,
         update_target_rate=args.target_rate,
         render=args.render,
         test=args.test,
         log_dir=args.log_dir,
         save_weights_episode=args.save_weights_episode)
