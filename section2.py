import numpy as np
from section1 import Domain
from functools import lru_cache

class agent:
    def __init__(self):
        self.actions = [(1, 0), (-1, 0), (0, 1), (0, -1)] # actions the agent can choose from

    def chose_action(self, state):
        """Returns an action given a state"""
        return self.actions[np.random.choice(len(domain.actions))]
    
    def left(self):
        """Returns the action left"""
        return (-1, 0)
    
if __name__ == "__main__":
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
    agent = agent()

    # J function for the deterministic domain
    @lru_cache(maxsize=None)
    def det_j_func(domain, state, agent, N):
        if N == 0:
            return 0
        else:
            a = agent.left()
            j = domain.det_reward(state, a) + domain.discount * det_j_func(domain, domain.dynamic(state, a), agent, N-1)
            return j
        
    # J function for the stochastic domain
    @lru_cache(maxsize=None)
    def sto_j_func(domain, state, agent, N):
        if N == 0:
            return 0
        else:
            a = agent.left()
            j = 0.5*(domain.det_reward(state, a) + domain.discount * sto_j_func(domain, domain.dynamic(state, a), agent, N-1))
            j += 0.5*(domain.g[0, 0] + domain.discount * sto_j_func(domain, (0, 0), agent, N-1))
            return j
        
    # Estimate J for our random policy in deterministic domain:
    print('Deterministic domain')
    print("s; J(s)")
    for i in range(n):
        for j in range(m):
            s = (i,j)
            print(f"({i},{j}); {det_j_func(domain, s, agent, 300)}")

    # Estimate J for our random policy in stochastic domain:
    print('Stochastic domain')
    print("s; J(s)")
    for i in range(n):
        for j in range(m):
            s = (i,j)
            print(f"({i},{j}); {sto_j_func(domain, s, agent, 150)}")
