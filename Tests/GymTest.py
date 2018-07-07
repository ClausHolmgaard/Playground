import gym
import gym.spaces 

#from gym import envs
#print(envs.registry.all())

env = gym.make('CartPole-v0')
#env = gym.make('MsPacman-v0')
#env = gym.make('Hopper-v2')
env.reset()
for _ in range(1000):
    env.render()
    action = env.action_space.sample() # random action
    observation, reward, done, info = env.step(action)
#    if done:
#      print("Done!")
#      break
env.close()

