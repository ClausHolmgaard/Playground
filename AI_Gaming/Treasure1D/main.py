# See: https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/1_command_line_reinforcement_learning/treasure_on_right.py

import time

from Treasure1DAgent import *
from Treasure1DEnvironment import Environment

environment = Environment(num_state=15)
AgentObject = TDAgent
agent = AgentObject(environment.get_states(),
                    environment.get_actions())

MAX_ITERATIONS_TRAIN = 100
MAX_ITERATIONS_VIS = 20
EPISODES = 100


#training
for i in range(EPISODES):
    s = environment.reset()
    agent.reset_after_episode()
    environment.visualize()
    done = False
    ite = 0

    while not done:
        a = agent.action(s)
        r, s, done = environment.reward_state(a)
        environment.visualize()
        agent.reward(r)

        print(f" Iteration {i}", end='')

        if done:
            print(f" Goal reached!")

        ite += 1
        if ite == MAX_ITERATIONS_TRAIN:
            print(" Goal NOT reached.")
            break

        time.sleep(0.1)

"""
print("Training done, visualizing...")
# Visualize
for i in range(5):
    s = environment.reset()
    environment.visualize()
    done = False
    ite = 0

    while not done:
        a = agent.action(s)
        r, s, done = environment.reward_state(a)
        environment.visualize()

        if done:
            print(f" Goal reached!")

        ite += 1
        if ite >= MAX_ITERATIONS_VIS:
            print(" Goal NOT reached.")
            break
        
        time.sleep(0.1)
"""
print()
agent.final_report()
