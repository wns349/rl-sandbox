import gym
import matplotlib.pyplot as plt
import numpy as np

EPISODES = 5000


def main():
    env = gym.make("CartPole-v1")

    scores = []
    for episode in range(EPISODES):
        env.reset()
        score = 0
        while True:
            action = env.action_space.sample()
            next_observation, reward, is_done, _ = env.step(action)
            score += reward
            if is_done:
                print("Game over! score: {}".format(score))
                break
        scores.append(score)

    plt.plot(scores)
    plt.xlabel("Episodes")
    plt.ylabel("Scores")
    plt.savefig("./cartpole_random.png")
    plt.show()

    print("Average score: ", np.mean(scores))

    env.close()


if __name__ == '__main__':
    main()
