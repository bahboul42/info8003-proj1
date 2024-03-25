from section1 import Domain, MomentumAgent, AcceleratingAgent
from display_caronhill import save_caronthehill_image
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio

def make_video(trajectory):
    filenames = []
    if not os.path.exists("gif"):
        os.makedirs("gif")
    t = str(time.time())
    if not os.path.exists(t):
        os.makedirs(f"gif/{t}")
    for i, var in enumerate(trajectory):
        print(f"Step {i+1}/{len(trajectory)}", flush=True)
        print(var)
        p, s, _, _, _, _ = var
        try :
            filenames.append(f"gif/{t}/caronhill_{i+1}.png")
            save_caronthehill_image(p, s, out_file=f"gif/{t}/caronhill_{i+1}.png") # Save the image
        except KeyboardInterrupt:
            break
    
    print("Creating gif...")
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(f'gif/{t}/movie.gif', images, duration=1e-7)

    print("Removing images...")
    for filename in filenames:
        os.remove(filename)

    print("Done!")
    pass

if __name__ == "__main__":

    domain = Domain() # Create the environment
    state = domain.set_state(0, 0) # Sample an initial state

    agent = AcceleratingAgent() # Create the agent
    
    n_steps = 1000
    for _ in range(n_steps):
        state = domain.get_state()
        action = agent.get_action(state)
        domain.step(action)

    traj = domain.get_trajectory()
    p = [x[0] for x in traj]
    s = [x[1] for x in traj]

    plt.plot(np.arange(len(p)), p, label='p')
    plt.plot(np.arange(len(s)), s, label='s')
    plt.legend()
    plt.show()

    make_video(traj)

