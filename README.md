# Reinforcement Learning in a Discrete Domain - INFO8003-1

## Project 1: Optimal Decision Making for Complex Problems

This project is part of the INFO8003-1 course, "Optimal Decision Making for Complex Problems," offered at the University of Liège under the guidance of Prof. Damien Ernst. This effort is a collaboration between Andreas Coco and Guillaume Delporte, undertaken with the support of teaching assistants Arthur Louette and Bardhyl Miftari.

### Overview

The goal of this project is to explore and implement various reinforcement learning techniques within a defined discrete domain. The domain consists of a grid where an agent can move and receive rewards based on its actions. We aim to analyze both deterministic and stochastic dynamics within this environment, employing reinforcement learning strategies to navigate the complexities presented.

### Domain Description

- **State Space (S)**: A set of tuples \((x, y) \in \mathbb{N}^2\) where \(x < n\) and \(y < m\).
- **Action Space (A)**: Four possible movements represented by the tuples \((1, 0)\), \((-1, 0)\), \((0, 1)\), and \((0, -1)\).
- **Reward Signal**: A function \(r((x, y), (i, j))\) that returns a reward based on the current position and action.
- **Discount Factor (\(\gamma\))**: 0.99.
- **Time Horizon (T)**: Approaches infinity.

The project explores two distinct domains: a deterministic domain where outcomes are predictable and a stochastic domain where outcomes are affected by randomness.

### Project Structure

#### 1. Implementation of the Domain

- Implementation details of the domain based on a specific instance depicted in the project description.
- A rule-based policy simulation showing trajectories over 10 steps from an initial state.

#### 2. Expected Return of a Policy

- Calculation of the expected return of a stationary policy over an infinite time horizon.
- Results displayed for each state in a structured table format.

#### 3. Optimal Policy

- Computation of state transition probabilities and rewards.
- Determination of the optimal policy using dynamic programming techniques.

#### 4. System Identification

- Estimation of transition probabilities and rewards from given trajectories.
- Analysis of convergence speeds and approximation quality.

#### 5. Q-Learning in a Batch Setting

- Implementation and comparison of offline and online Q-learning methods.
- Exploration of different learning rates and the impact of a replay buffer on learning efficiency.

### Experimental Protocols

The project includes various experimental protocols designed to evaluate the performance of reinforcement learning algorithms under different conditions. These protocols involve offline Q-learning, online Q-learning with different learning rates, and the use of a replay buffer.

### Discussion and Results

The report discusses the implementation details, experimental results, and the implications of the findings. Even in the absence of expected results, the discussion focuses on the potential reasons and theoretical considerations behind the observed behaviors.

### Bonus: Policy Optimization Algorithm - SARSA

As an additional challenge, we explore the SARSA algorithm, comparing it with online Q-learning and discussing its unique characteristics and performance in our domain.

### How to Run

The source code is organized into Python files corresponding to each section of the project. To execute the code:

1. Ensure Python 3.10 is installed along with NumPy and matplotlib.
2. Run each section's file separately, following the naming convention `sectionK.py` where `K` is the section number.

### Contributors

- Andreas Coco
- Guillaume Delporte

Under the supervision of Prof. Damien Ernst, and assistants Arthur Louette and Bardhyl Miftari.

### Contact

For any inquiries related to this project, please contact the contributors via GitHub.

