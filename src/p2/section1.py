import numpy as np

class Domain:
    def __init__(self, g=9.81, int_time_step=0.001, dis_time_step=0.100, action_space=[4, -4], m=1, discount=0.95):
        self.g = g  # Gravitational acceleration
        self.int_time_step = int_time_step  # Integration time step for Euler method
        self.dis_time_step = dis_time_step  # Discretization time step
        self.action_space = action_space  # Possible actions
        self.m = m
        self.discount = discount  # Discount factor

        self.p = None  # Position
        self.s = None

        self.trajectory = []  # Trajectory of states and actions

    def reset(self):
        """Resets the environment."""
        self.sample_initial_state()
        self.trajectory = []

    def sample_initial_state(self):
        """Samples an initial state from the initial state distribution."""
        self.p = np.random.uniform(-0.1, 0.1)
        self.s = 0

    def get_state(self):
        """Returns the current state of the environment."""
        return self.p, self.s
    
    def set_state(self, p, s):
        """Sets the state of the environment."""
        self.p = p
        self.s = s
    
    def update_trajectory(self, x, a, r, x_prime):
        """Updates the trajectory with the current state and action."""
        p, s = x
        p_prime, s_prime = x_prime
        self.trajectory.append((p, s, a, r, p_prime, s_prime))

    def reset_trajectory(self):
        """Resets the trajectory and sample new initial state."""
        self.trajectory = []

    
    def get_trajectory(self):
        """Returns the trajectory."""
        return self.trajectory
    
    def hill(self, p, derivative=0):
        """Defines the Hill function based on the position p."""
        if derivative not in [0, 1, 2]:
            raise ValueError("Invalid derivative order.")
        
        if derivative == 0:
            if p < 0:
                return (p**2 + p)
            else:
                return p / np.sqrt(1 + 5*p**2)
        elif derivative == 1:
            if p < 0:
                return 2*p + 1
            else:
                return 1/(5*p**2+1)**(3/2)
        else :
            if p < 0:
                return 2
            else:
                return -15*p/(5*p**2+1)**(5/2)
            
    def dynamics(self, p, s, u):
        """Computes the next state given the current state and action using Euler's method."""
        dp = s
        ds = u / self.m*(1 + self.hill(p, 1)**2) - (self.g * self.hill(p)) / (1 + self.hill(p, 1)**2) - (s**2 * self.hill(p, 1) * self.hill(p, 2)) / (1 + self.hill(p, 1)**2)
        p_next = p + dp * self.int_time_step
        s_next = s + ds * self.int_time_step 
        if abs(p_next) > 1 or abs(s_next) > 3: # not sure if this is correct : reward = 0 after/ snapping car to 1?
            return p_next, 0
        return p_next, s_next

    def reward(self, p_next, s_next):
        """Computes the reward based on the next state."""
        if p_next < -1 or abs(s_next) > 3:
            return -1
        elif p_next > 1 and abs(s_next) <= 3:
            return 1
        else:
            return 0

    def step(self, action):
        """Takes a step in the environment."""
        p, s = self.get_state()
        for _ in range(int(self.dis_time_step / self.int_time_step)): # Discretization
            p_next, s_next = self.dynamics(p, s, action)
            self.set_state(p_next, s_next)
        r = self.reward(p_next, s_next)
        self.update_trajectory((p, s), action, r, (p_next, s_next))
        return (p, s), action, r, (p_next, s_next)
    
    def print_trajectory(self):
        """Prints the trajectory."""
        traj = self.get_trajectory()
        l_traj = len(traj)

        print('Step (p, s), a, r, (p_prime, s_prime)')
        for i, t in enumerate(traj):
            p, s, a, r, p_prime, s_prime = t
            print(f"Step ({i+1}/{l_traj} : {p}, {s}), {a}, {r}, ({p_prime}, {s_prime})", flush=True)

class AcceleratingAgent:
    def __init__(self):
        pass

    def get_action(self, state):
        return 4

class MomentumAgent:
    def __init__(self):
        self.direction = -1

    def get_action(self, state):
        _, s = state
        if s == 0:
            self.direction *= -1
        return 4 * self.direction # problem when s = 0

if __name__ == "__main__":

    domain = Domain() # Create the environment
    domain.sample_initial_state() # Sample an initial state
    
    agent = MomentumAgent() # Create the agent

    # Simulate the system
    n_steps = 1000
    for _ in range(n_steps):
        state = domain.get_state()
        action = agent.get_action(state)
        domain.step(action)

    domain.print_trajectory()

    domain.reset() # Reset the trajectory for future use