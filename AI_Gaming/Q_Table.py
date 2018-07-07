import numpy as np
import matplotlib.pyplot as plt
import gym
import gym.spaces

env = gym.make('FrozenLake-v0')
#env.reset()

print("Possible states: {}".format(env.observation_space.n))
print("Possible actions: {}".format(env.action_space.n))

Q = np.zeros([env.observation_space.n, env.action_space.n])

lr = 0.8
y = 0.95
num_episodes = 2000
num_tests = 1000
noise_chance = 0.25

dbg_times_rewarded = 0

for i in range(num_episodes):
    if i % 500 == 0:
        print("Episode: {}".format(i))

    s = env.reset()
    counter = 0
    done = False
    max_moves = 20

    while done is False:
        counter += 1

        """
        # Q action
        a_q = np.argmax(Q[s, :])
        # Random action
        a_r = np.random.randint(0, 4)

        if np.all(Q[s, :] == 0):
            a = a_r
        else:
            a = np.random.choice([a_q, a_r], p=[1 - noise_chance, noise_chance])
        """

        #a = np.argmax(Q[s, :] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))
        a = np.argmax(Q[s, :] + np.random.randn(1, env.action_space.n) * 0.005)
        #a = np.argmax(Q[s, :])

        #Get new state and reward from environment
        s1, r, done, _ = env.step(a)
        #Update Q-Table with new knowledge

        if r == 1:
            dbg_times_rewarded += 1

        #if done and r == 0:
        #    r = -1

        #if counter > max_moves:
        #    r = -1
        #    done = True

        Q[s, a] = Q[s, a] + lr * (r + y * np.max(Q[s1, :]) - Q[s, a])
        #Q[s, a] = r + y * np.max(Q[s1, :])
        s = s1

        #if np.max(Q) - np.min(Q) != 0:
        #    Q = (Q - np.min(Q)) / (np.max(Q) - np.min(Q))

        if done == True:
            break


print("Final Q table: ")
print(Q)

print()
print("Times rewarded: {}".format(dbg_times_rewarded))
print("Max Q: {}, min Q: {}".format(np.max(Q), np.min(Q)))
print("Done training!")
print()

# Let's test it
test_success = 0
for i in range(num_tests):
    s = env.reset()
    counter = 0

    while counter < 99:
        counter += 1

        a = np.argmax(Q[s, :])
        s1, r, done, _ = env.step(a)

        if r == 1:
            test_success += 1

        s = s1

        if done:
            break

print("Test success: {}".format(test_success))

env.close()


"""
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    if done:
        print("Done!")
        break
"""


