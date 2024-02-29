import numpy as np
from section1 import Domain
from functools import lru_cache
import sys

class Agent:
    def __init__(self, domain, policy=(0.25, 0.25, 0.25, 0.25)):
        self.actions = domain.actions # Actions the agent can execute
        self.policy = policy # Policy the agent follows

if __name__ == "__main__":
    # Increase the recursion limit
    print(sys.setrecursionlimit(10**5))

    n, m = 5, 5  # Grid size
    # Rewards
    g = np.array([[-3,   1,  -5, 0,  19],
                  [ 6,   3,   8, 9,  10],
                  [ 5,  -8,   4, 1,  -8],
                  [ 6,  -9,   4, 19, -5],
                  [-20, -17, -4, -3,  9]])

    s0 = (3, 0)  # Initial state

    # Initialize the domain
    domain = Domain(n, m, g, s0, random_state=42)

    # Initialize the agent
    agent = Agent(domain)

    @lru_cache(maxsize=None) # Use the cache to avoid recomputing several times the same values
    def det_j_func(domain, state, agent, N):
        """
        Calculates the value function J_N for a deterministic domain using the agent's policy
        """
        if N == 0:
            return 0
        else:
            j = 0
            for i in range(len(agent.actions)):
                j += agent.policy[i] * (domain.det_reward(state, agent.actions[i]) + domain.discount * det_j_func(domain, domain.dynamic(state, agent.actions[i]), agent, N-1))
            return j
        
    @lru_cache(maxsize=None) # Use the cache to avoid recomputing several times the same values
    def sto_j_func(domain, state, agent, N):
        """
        Calculates the value function J_N for a stochastic domain using the agent's policy
        """
        if N == 0:
            return 0
        else:
            j = 0
            for i in range(len(agent.actions)):
                j += agent.policy[i] * (domain.det_reward(state, agent.actions[i]) + domain.discount * sto_j_func(domain, domain.dynamic(state, agent.actions[i]), agent, N-1))
            j = 0.5 * j
            j += 0.5 * (domain.g[0, 0] + domain.discount * sto_j_func(domain, (0, 0), agent, N-1))
            return j
        
    # Value of N used to estimate J
    N = 10**4

    # Estimate J for the random policy in the deterministic domain
    print('Deterministic domain')
    print("s; J(s)")
    for i in range(n):
        for j in range(m):
            s = (i,j)
            print(f"({i},{j}); {det_j_func(domain, s, agent, N)}")

    # Estimate J for the random policy in the stochastic domain
    print('Stochastic domain')
    print("s; J(s)")
    for i in range(n):
        for j in range(m):
            s = (i,j)
            print(f"({i},{j}); {sto_j_func(domain, s, agent, N)}")

    # Estimate the accuracy of the estimation
    print("Estimation accuracy:")
    print((domain.discount**(N))*np.max(domain.g)/(1-domain.discount))