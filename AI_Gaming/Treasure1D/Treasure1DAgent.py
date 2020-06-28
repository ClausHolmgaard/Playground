import numpy as np
import random

class BaseAgent(object):
    def __init__(self, states, actions):
        self.possible_states = states
        self.possible_actions = actions
    
    def action(self, state):
        raise NotImplementedError
    
    def reward(self, reward):
        raise NotImplementedError

    def final_report(self):
        raise NotImplementedError

class RandomAgent(BaseAgent):
    def action(self, state):
        action = random.choice(self.possible_actions)
        return action
    
    def final_report(self):
        pass

class TDAgent(BaseAgent):
    def __init__(self, possible_states, possible_actions, td_step=0):
        self.possible_actions = possible_actions
        self.possible_states = possible_states
        self.td_step = td_step

        self.alfa = 0.1
        self.gamma = 0.9

        num_states = len(self.possible_states)
        num_actions = len(self.possible_actions)

        self.q_table = np.zeros((num_states, num_actions))

        self.reset_after_episode()
    
    def reset_after_episode(self):
        history_length = self.td_step + 2

        # holding a hostory of state, action, reward
        self.state_action_history = np.zeros((history_length, 3), dtype=np.uint8)
        self.moves_made = 0

    def policy(self, state):
        action = np.argmax(self.q_table[state])
        return action

    def action(self, state):
        action = self.policy(state)

        # The '0' is a placeholder, will be overridden when reward is received
        self.history_add([state, action, 0])
        self.moves_made += 1
        return self.possible_actions[action]

    def reward(self, reward_given):
        self.state_action_history[-1][2] = reward_given

        if self.moves_made < self.td_step + 2:
            return
        
        prev_state = self.state_action_history[1][0]
        prev_action = self.state_action_history[1][1]
        this_state = self.state_action_history[0][0]
        this_action = self.state_action_history[0][1]

        #G = reward_given
        #for i in range(1, self.td_step + 1):
        #    G += self.gamma**i * self.state_action_history[-i][2]

        
        update = reward_given + \
                 self.gamma * self.q_table[this_state, this_action] - \
                 self.q_table[prev_state, prev_action]

        self.q_table[prev_state, prev_action] = self.q_table[prev_state, prev_action] + self.alfa * update
        

    def history_add(self, value):
        v = np.array(value, dtype=np.uint8)
        self.state_action_history[1:] = self.state_action_history[:-1]
        self.state_action_history[0] = v

    def final_report(self):
        print(self.q_table)

if __name__ == "__main__":
    pa = ['L', 'R']
    ps = ['-'] * (6 - 1) + ['T']
    a = TDOneStepAgent(ps, pa)
    
    ac = a.action(4)
    print(ac)
    a.reward(-1)
    a.final_report()

    print()
    ac = a.action(3)
    print(ac)
    a.reward(-1)
    a.final_report()

    print()
    ac = a.action(2)
    print(ac)
    a.reward(-1)
    a.final_report()
    
    print()
    ac = a.action(1)
    print(ac)
    a.reward(-1)
    a.final_report()
    
    print()
    ac = a.action(0)
    print(ac)
    a.reward(-1)
    a.final_report()

    print()
    ac = a.action(0)
    print(ac)
    a.reward(-1)
    a.final_report()
    
    