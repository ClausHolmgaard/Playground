import numpy as np

REWARD_ON_INVALID_MOVE = -1
REWARD_ON_GOAL = 100
REWARD_ON_MOVE = -1

ACTIONS = ['L', 'R']

class Environment(object):
    def __init__(self, num_state=10):
        self.num_states = num_state
        self.reset()
        self.done = False
    
    def get_states(self):
        return self.states
    
    def get_actions(self):
        return ACTIONS

    def reset(self, state=None):
        self.done = False
        self.states = ['-'] * (self.num_states - 1) + ['T']   # '---------T' our environment
        if state is None:
            self.state = np.random.randint(0, len(self.states) - 1) # subtracting one, excluding goal
        else:
            self.state = state
        return self.state

    def reward_state(self, action):
        if self.done:
            return None, None, True

        reward = 0

        # Actions: 'L' or 'R'
        if action == 'L':
            if self.state > 0:
                self.state -= 1
                reward = REWARD_ON_MOVE
            elif self.state == 0:
                reward = REWARD_ON_INVALID_MOVE

        if action == 'R':
            if self.state < len(self.states) - 1:
                self.state += 1
                reward = REWARD_ON_MOVE
        
        if self.check_goal():
            reward = REWARD_ON_GOAL
            self.done = True

        return reward, self.state, self.done
    
    def check_goal(self):
        return self.states[self.state] == 'T'
    
    def visualize(self):
        v = self.states.copy()
        v[self.state] = 'o'

        print(f"\r{' ' * 20}", end='')
        print(f"\r{''.join(v)}", end='')

if __name__ == "__main__":
    print()
    e = Environment()
    e.reset()

    import time
    for _ in range(10):
        r, s, d = e.reward_state('R')
        e.visualize()
        if d:
            break
        time.sleep(1)
    
    print()
