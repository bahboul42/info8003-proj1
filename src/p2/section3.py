from section1 import Domain, MomentumAgent

def make_video():

    pass

if __name__ == "__main__":

    domain = Domain() # Create the environment
    state = domain.set_state(0, 0) # Sample an initial state

    agent = MomentumAgent() # Create the agent
    
    n_steps = 1000
    for _ in range(n_steps):
        state = domain.get_state()
        action = agent.get_action(state)
        domain.step(action)

    traj = domain.get_trajectory()
    make_video(traj)

