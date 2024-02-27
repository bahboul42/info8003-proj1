import numpy as np
from section1 import Domain
from functools import lru_cache

import sys



class Agent:
    def __init__(self, domain, policy = (0.25, 0.25, 0.25, 0.25)):
        self.actions = domain.actions # actions the agent can choose from
        self.policy = policy # probability of choosing each action

    def chose_action(self, state):
        """Returns an action given a state"""
        which_action = np.random.rand()
        if which_action <= self.policy[0]:
            return self.actions[0]
        elif which_action <= self.policy[0] + self.policy[1]:
            return self.actions[1]
        elif which_action <= self.policy[0] + self.policy[1] + self.policy[2]:
            return self.actions[2]
        else:
            self.actions[3]
        
    
if __name__ == "__main__":
    print(sys.setrecursionlimit(10**5))

    n, m = 5, 5  # Grid size
    g = np.array([[-3,   1,  -5, 0,  19],
                  [ 6,   3,   8, 9,  10],
                  [ 5,  -8,   4, 1,  -8],
                  [ 6,  -9,   4, 19, -5],
                  [-20, -17, -4, -3,  9],])

    s0 = (3, 0)  # Initial state

    # Initialize the domain
    domain = Domain(n, m, g, s0, random_state=42)

    # Initialize the agent
    agent = Agent(domain)

    # J function for the deterministic domain
    @lru_cache(maxsize=None)
    def det_j_func(domain, state, agent, N):
        if N == 0:
            return 0
        else:
            j = 0
            for i in range(len(agent.actions)):
                j += agent.policy[i] * (domain.det_reward(state, agent.actions[i]) + domain.discount * det_j_func(domain, domain.dynamic(state, agent.actions[i]), agent, N-1))
            return j
        
    # J function for the stochastic domain
    @lru_cache(maxsize=None)
    def sto_j_func(domain, state, agent, N):
        if N == 0:
            return 0
        else:
            j = 0
            for i in range(len(agent.actions)):
                j += agent.policy[i] * (domain.det_reward(state, agent.actions[i]) + domain.discount * sto_j_func(domain, domain.dynamic(state, agent.actions[i]), agent, N-1))
            j = 0.5 * j
            j += 0.5 * (domain.g[0, 0] + domain.discount * sto_j_func(domain, (0, 0), agent, N-1))
            return j
        
    # Value of N we use to estimate J
    N = 10**4

    # Estimate J for our random policy in deterministic domain:
    print('Deterministic domain')
    print("s; J(s)")
    for i in range(n):
        for j in range(m):
            s = (i,j)
            print(f"({i},{j}); {det_j_func(domain, s, agent, N)}")

    # Estimate J for our random policy in stochastic domain:
    print('Stochastic domain')
    print("s; J(s)")
    for i in range(n):
        for j in range(m):
            s = (i,j)
            print(f"({i},{j}); {sto_j_func(domain, s, agent, N)}")

    print("Estimation accuracy:")
    print((domain.discount**(N))*np.max(domain.g)/(1-domain.discount))

