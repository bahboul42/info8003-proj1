from section1 import Domain, MomentumAgent
from display_caronhill import save_caronthehill_image
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def make_video(trajectory):
    filenames = []
    if not os.path.exists("gif"):
        os.makedirs("gif")
    t = str(time.time())
    gif_path = f"gif/{t}"
    if not os.path.exists(gif_path):
        os.makedirs(gif_path)
    for i, var in enumerate(trajectory):
        print(f"Step {i+1}/{len(trajectory)}", flush=True)
        print(var)
        p, s, _, _, _, _ = var
        try:
            filename = f"{gif_path}/caronhill_{i+1}.png"
            filenames.append(filename)
            save_caronthehill_image(p, s, out_file=filename)  
        except KeyboardInterrupt:
            break

    print("Creating gif...")
    images = []
    for filename in filenames:
        with Image.open(filename) as img:
            images.append(img.copy())
    gif_filename = f'{gif_path}/movie.gif'
    images[0].save(gif_filename, save_all=True, append_images=images[1:], duration=1.67, loop=0)

    print("Removing images...")
    for filename in filenames:
        os.remove(filename)

    print("Done!")

if __name__ == "__main__":

    domain = Domain() # Create the environment
    state = domain.set_state(0, 0) # Sample an initial state

    agent = MomentumAgent() # Create the agent
    
    n_steps = 3000
    for _ in range(n_steps):
        state = domain.get_state()
        action = agent.get_action(state)
        domain.step(action)

    traj = domain.get_trajectory()
    p = [x[0] for x in traj]
    s = [x[1] for x in traj]

    # plt.figure(figsize=(10, 6))
    # plt.plot(np.arange(len(p)), p, label='p')
    # plt.plot(np.arange(len(s)), s, label='s')
    # plt.legend()
    # plt.show()

    make_video(traj)
