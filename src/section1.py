import numpy as np

class Domain:
    def __init__(self, n, m, g, initial_state, random_state=None):
        self.n = n  # Grid size in x
        self.m = m  # Grid size in y
        self.g = g  # Reward grid

        self.current_x = initial_state[0]
        self.current_y = initial_state[1]

        self.discount = .99

        np.random.seed(random_state)
        self.w = np.random.rand()

        self.actions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # Down, Up, Right, Left

    def get_current_state(self):
        """Return the current state."""
        return self.current_x, self.current_y
    
    def set_state(self, state):
        """Set the current state."""
        self.current_x, self.current_y = state

    def det_reward(self, state, action):
        """Return the reward for a given state."""
        x, y = self.det_dyn(state, action)
        return self.g[x, y]
    
    def sto_reward(self, state, action):
        """Return the reward for a given state."""
        if self.w <= 0.5: # do not draw a new random number > only when step is taken
            x, y =  self.dynamic(state, action)
            return self.g[x, y]
        else:
            x, y = (0, 0)
            return self.g[x, y]
    
    def det_step(self, action):
        """Deterministic step."""
        new_state = self.det_dyn(self.get_current_state(), action)
        last_state = self.get_current_state()
        self.current_x = new_state[0]
        self.current_y = new_state[1]
        return last_state, action,  self.det_reward(last_state, action), new_state
    
    def sto_step(self, action):
        """Stochastic step."""
        last_state = self.get_current_state()
        obtained_reward = self.sto_reward(last_state, action)
        new_state = self.sto_dyn(self.get_current_state(), action)
        self.current_x = new_state[0]
        self.current_y = new_state[1]
        return last_state, action,  obtained_reward, new_state

    def dynamic(self, state, action):
        """Compute the next state given current state and action."""
        x, y = state
        i, j = action
        new_x = min(max(x + i, 0), self.n - 1)
        new_y = min(max(y + j, 0), self.m - 1)
        return (new_x, new_y)
    
    def det_dyn(self, state, action):
        """Deterministic dynamics."""
        return self.dynamic(state, action)
    
    def sto_dyn(self, state, action):
        """Stochastic dynamics."""
        if self.w <= 0.5:
            self.w = np.random.rand()
            return self.dynamic(state, action)
        else:
            self.w = np.random.rand()
            return (0, 0)
        



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

    # Rule-based policy: Select a move at random
    def policy(state):
        return domain.actions[np.random.choice(len(domain.actions))]
    
    # # Deterministic steps
    print("Deterministic trajectory")
    print("s_t,  a_t,  r_t,  s_(t+1)")
    current_state = s0
    for i in range(11):
        current_action = policy(current_state)
        print(domain.det_step(current_action))

    # Restart at the initial state
    domain.set_state((3, 0))
    
    # # Stochastic steps
    print("Stochastic trajectory")
    print("s_t,  a_t,  r_t,  s_(t+1)")
    current_state = s0
    for i in range(11):
        current_action = policy(current_state)
        print(domain.sto_step(current_action))
    
