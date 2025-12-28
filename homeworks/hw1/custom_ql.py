import numpy as np
from collections import defaultdict


class CustomQLAgent:
    def __init__(self, starting_state, state_space, action_space, alpha=0.1, gamma=0.99, 
                 initial_epsilon=0.05, min_epsilon=0.005, epsilon_decay=1):

        self.state = starting_state
        self.action = None
        
        self.state_space = state_space
        self.action_space = action_space

        self.alpha = alpha
        self.gamma = gamma

        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay

        self.Q = defaultdict(lambda: np.zeros(self.action_space.n))

    def act(self):
        if np.random.rand() < self.epsilon:
            self.action = np.random.randint(self.action_space.n)
        else:
            self.action = np.argmax(self.Q[self.state])
            
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        return self.action

    def learn(self, next_state, reward):
        Q_sa = self.Q[self.state][self.action]
        V_sn = np.max(self.Q[next_state])

        td_error = reward + self.gamma * V_sn - Q_sa
        self.Q[self.state][self.action] += self.alpha * td_error

        self.state = next_state

