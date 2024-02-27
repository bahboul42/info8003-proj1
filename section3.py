from section1 import Domain
from section2 import Agent
import numpy as np
from itertools import product



def det_proba(domain):
    probas = np.zeros((domain.g.size, domain.g.size, len(domain.actions)))

    for s_prime, s, a in product(range(domain.g.size), range(domain.g.size), range(len(domain.actions))):
        action = domain.actions[a]
        state = (s // domain.n, s % domain.n)
        state_prime = domain.dynamic(state, action)

        if s_prime == state_prime[0] * domain.n + state_prime[1]:
            probas[s_prime, s, a] = 1
    return probas

def sto_proba(domain):
    probas = np.zeros((domain.g.size, domain.g.size, len(domain.actions)))

    for s_prime, s, a in product(range(domain.g.size), range(domain.g.size), range(len(domain.actions))):
        if s_prime == 0:
            probas[s_prime, s, a] += 0.5
        
        action = domain.actions[a]
        state = (s // domain.n, s % domain.n)
        state_prime = domain.dynamic(state, action)

        if s_prime == state_prime[0] * domain.n + state_prime[1]:
            probas[s_prime, s, a] += 0.5
    return probas

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







