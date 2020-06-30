import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))

    def select_action(self, state, eps = .5):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        if np.random.random() > eps:
            return np.argmax(self.Q[state])
        else:
            return np.random.choice(np.arange(self.nA))

    def step(self, state, action, reward, next_state, done, alpha = .2, gamma = 1, eps = .5):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        if not done:
            this_V = self.Q[state][action]
            next_action = self.select_action(next_state,eps)
            prob_s = np.ones(self.nA)*eps/self.nA
            prob_s[np.argmax(self.Q[next_state])] = 1 - eps + eps/self.nA
            self.Q[state][action] = this_V + alpha*(reward + gamma*np.dot(self.Q[next_state], prob_s) - this_V)

        else:
            self.Q[state][action] = self.Q[state][action] + alpha*(reward - self.Q[state][action])
