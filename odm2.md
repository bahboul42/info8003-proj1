# Rappels Optimal Decision Making for Complex Problems
## Ernst
### Key Points from Damien Ernst's Paper
**Stopping Conditions:**
1. **Maximum Number of Iterations:** 
   - The process can be predefined to stop after a certain number of iterations. This number can be determined by ensuring that the error bound on the sub-optimality of the policy falls below a pre-defined tolerance level.
   - The error bound equation: $ \|J_{\mu^*_N} - J_{\mu^*}\|_\infty \leq \frac{2 \gamma^N B_r}{(1-\gamma)^2} $
   - Here, $ B_r $ is a known constant, and $ \gamma $ is the discount factor.

2. **Distance Between Successive Approximations:**
   - Another method is to stop when the distance between successive $ Q $-functions $ \| \hat{Q}_N - \hat{Q}_{N-1} \| $ falls below a certain threshold. 
   - Note: This method assumes the supervised learning algorithm guarantees convergence, which is not always the case in practice.

#### Key Points to Know for Oral Exam

1. **Basic Concepts:**
   - Understand the fundamentals of reinforcement learning, including policies, value functions, and the Bellman equation.
   - Know the difference between model-free and model-based methods.
   
2. **Fitted Q Iteration:**
   - The core algorithm discussed in the paper.
   - It involves iteratively improving the Q-function using regression techniques on a dataset of observed transitions.

3. **Stopping Conditions:**
   - Clearly explain the stopping conditions as outlined above.

4. **Performance Evaluation:**
   - Discuss how the quality of the computed policy is evaluated. Mention the use of the number of successful trajectories and score values to assess performance.
   - Highlight the importance of the reward function and its parameters (e.g., $ c_{reward} $ and $ \gamma $) in determining the success of a policy.

5. **Artificial Trajectories:**
   - Explain how artificial trajectories can be generated to improve the learning process.
   - The process involves using existing data points to create new transitions that help in better approximating the Q-function.

6. **Convergence and Consistency:**
   - Understand the theoretical aspects regarding the convergence of the fitted Q iteration algorithm.
   - Mention any assumptions made for the convergence and discuss the practical implications if these assumptions do not hold.

7. **Practical Implementation:**
   - Be familiar with the Extra-Trees algorithm used for regression in the context of fitted Q iteration.
   - Discuss the computational complexity and strategies to handle large datasets.

8. **Key Algorithms and Methods:**
   - Extra-Trees: Know its parameters (e.g., $ n_{min}, K $) and how it builds decision trees for regression.
   - Other tree-based methods mentioned in the paper, such as Tree Bagging and Totally Randomized Trees.

9. **Case Studies and Experiments:**
   - Be prepared to discuss specific control problems used as benchmarks in the experiments, like the “Bicycle Balancing” problem.
   - Highlight the results and insights gained from these experiments, including how different configurations of the reward function and discount factor affect performance.

## Raphael
## Low-data Reinforcement Learning: working with (almost) no data and no model
### Conditions de Lipschitz
Lipschitz continuity is an important assumption in reinforcement learning to ensure stability and convergence. In the context of the document, the functions $ f $ (system dynamics), $ \rho $ (reward function), and $ h $ (policy) are assumed to be Lipschitz continuous. This means there exist constants $ L_f $, $ L_\rho $, and $ L_h $ such that for all states $ s, s' $ in the state space $ S $ and actions $ a, a' $ in the action space $ A $:

$$
\begin{align*}
\|f(s, a, w) - f(s', a', w)\|_S & \leq L_f (\|s - s'\|_S + \|a - a'\|_A), \\
|\rho(s, a, w) - \rho(s', a')| & \leq L_\rho (\|s - s'\|_S + \|a - a'\|_A), \\
\|h(t, s) - h(t, s')\|_A & \leq L_h \|s - s'\|_S,
\end{align*}
$$
for all $ (s, s', a, a', w) \in S^2 \times A^2 \times W $. This ensures that small changes in states or actions lead to proportionally small changes in the function values, which is crucial for the stability of the learning process.

### Reconstruire trajectoires artificielles
Artificial trajectories are synthesized sequences of state-action pairs constructed from samples of one-step transitions. The purpose of reconstructing artificial trajectories is to:

- Estimate the performance of policies.
- Compute performance guarantees.
- Develop safe policies.
- Generate additional transitions to enrich the training data.

These trajectories help in scenarios where direct simulations of the system are not possible or practical. They are built by minimizing the discrepancy with a classical Monte Carlo sample that could be obtained by simulating the system with the policy $ h $.

### Performance
In the context of reinforcement learning, performance typically refers to the expected return of a policy. The expected $ T $-stage return for a policy $ h $ starting from an initial state $ s_0 $ is defined as:

$$
J_h(s_0) = \mathbb{E} \left[ \sum_{t=0}^{T-1} \rho(s_t, h(t, s_t), w_t) \right],
$$

where the expectation is taken over the disturbances $ w_t $. Performance evaluation can also involve risk-sensitive criteria, considering the probability of the return falling below a certain threshold.

### Model-Free Monte Carlo Estimation
Model-free Monte Carlo (MFMC) estimation is an approach to estimate the performance of a policy without requiring a model of the system's dynamics. It mimics classical Monte Carlo estimation by reconstructing $ p $ artificial trajectories from samples of one-step transitions. The MFMC estimator for the expected return of policy $ h $ is given by:

$$
M_h^p(F_n, s_0) = \frac{1}{p} \sum_{i=1}^{p} \sum_{t=0}^{T-1} r_{i,t},
$$

where $ r_{i,t} $ are the rewards collected along the $ i $-th artificial trajectory. This estimator provides a way to evaluate policies using only the available batch data.

### Metrics
Metrics in reinforcement learning are used to evaluate the performance and reliability of policies. Common metrics include:

- **Expected Return**: The average cumulative reward received by following a policy.
- **Variance**: The variability in the returns, which indicates the stability of the policy.
- **Bias and Variance of Estimators**: Assess the accuracy and consistency of performance estimators like MFMC.

### What is a MFMC Estimator
A Model-Free Monte Carlo (MFMC) estimator is a method for evaluating the expected return of a policy by averaging the returns from multiple artificial trajectories. These trajectories are built from samples of one-step transitions to approximate the behavior of classical Monte Carlo estimation without needing a model of the system. The MFMC estimator helps to estimate the performance of a policy in batch mode reinforcement learning, especially when direct simulation of the system is not feasible.

These concepts are crucial for understanding the methodologies and assumptions underlying low-data batch mode reinforcement learning, as described in your document.

## Gaspard 
## Advanced algorithms for learning Q-functions
### What is an MDP?

A **Markov Decision Process (MDP)** is a mathematical framework for modeling decision-making where outcomes are partly random and partly under the control of a decision-maker. An MDP is defined by the tuple $ M = (S, A, T, R, p_0, \gamma) $:

- **States ($S$)**: The set of all possible states.
- **Actions ($A$)**: The set of all possible actions.
- **Transition Distribution ($T$)**: The probability distribution of transitioning from one state to another given an action, $ T(s_{t+1} | s_t, a_t) $.
- **Reward Function ($R$)**: The immediate reward received after transitioning from one state to another due to an action, $ R(s, a) $.
- **Initial Distribution ($p_0$)**: The distribution of the initial state $ s_0 $.
- **Discount Factor ($\gamma$)**: A factor between 0 and 1 that represents the importance of future rewards.

In MDPs, the Markov property holds, meaning the future state depends only on the current state and action, not on the history of past states and actions.

### MDP Execution

MDP execution refers to the process of interacting with the environment by following a policy to select actions based on states. The execution involves:

1. **History ($h_t$)**: The sequence of observed states and actions up to time $t$, $ h_t = (s_0, a_0, \ldots, s_t) $.
2. **Action Selection**: Selecting an action based on the current state or history. Ideally, actions depend on the history $h_t$, but by exploiting the Markov property, actions can depend on the current state $s_t$.

### Stochastic Policies

**Stochastic Policies** are policies that define a probability distribution over actions rather than selecting a single deterministic action. They are defined as:

- **History-Dependent Policy ($\eta$)**: A mapping from the history to a distribution over actions, $\eta(a_t | h_t)$.
- **Markov Policy ($\pi$)**: A mapping from the state to a distribution over actions, $\pi(a_t | s_t)$.

#### Reasons for Stochastic Policies:

1. **Exploration**: Ensures that the agent explores different actions to find the optimal policy.
2. **Handling Uncertainty**: Stochastic policies can better handle environments with inherent randomness.
3. **Robustness**: They can provide more robust solutions in uncertain and dynamic environments.

#### Why History Dependent:

History-dependent policies can utilize the entire history of states and actions, which might provide more information compared to just the current state. However, in practice, Markov policies are often used due to their simplicity and the optimality theorem, which states that there exists a Markov policy as good as any history-dependent policy.

#### Why Not Stationary:

Stationary policies are independent of time, which can be suboptimal in environments where the optimal action changes over time. Non-stationary policies can adapt to changing environments and dynamics.

### Théorèmes à connaître (Theorems to Know)

1. **Existence of the Optimal Policy**: There exists at least one history-dependent policy $\eta^*$ that maximizes the expected return for all states.
2. **Optimality of Markov Policies**: There exists at least one Markov policy that performs as well as the optimal history-dependent policy.
3. **Bellman Optimality Equation**: The optimal Q-function satisfies:
   $$
   Q(s, a) = R(s, a) + \gamma \mathbb{E}_{s'} \left[ \max_{a'} Q(s', a') \right]
   $$

### Bellman Equation

The Bellman equation relates the value of a state to the values of subsequent states. For the state-value function $V$ under policy $\pi$, it is:
$$
V^\pi(s) = \mathbb{E}_\pi \left[ R(s, a) + \gamma V^\pi(s') \right]
$$
For the Q-function:
$$
Q(s, a) = R(s, a) + \gamma \mathbb{E}_{s'} \left[ \max_{a'} Q(s', a') \right]
$$

### Target Network in Deep Q-Learning

A **Target Network** in deep Q-learning is a separate network used to stabilize the training of the Q-network. It is updated less frequently than the Q-network. The steps are:

1. Initialize the target network with the same weights as the Q-network.
2. Periodically update the target network weights with the Q-network weights.

**Minibatch**: A small, randomly sampled subset of the replay buffer used for training the Q-network. Minibatches help reduce the variance of gradient updates.

### Etapes de DeepQ

Steps in Deep Q-Learning:

1. **Initialize** the Q-network with random weights.
2. **Initialize** the target network with the same weights.
3. **Initialize** the replay buffer.
4. For each step:
   - Select an action using an $\epsilon$-greedy policy.
   - Execute the action and observe the reward and next state.
   - Store the transition in the replay buffer.
   - Sample a minibatch from the replay buffer.
   - Compute the target Q-value.
   - Perform a gradient descent step to minimize the loss between the predicted Q-values and target Q-values.
   - Periodically update the target network.

## Reinforcement learning in partially observable Markov decision processes
#### Markov Decision Process (MDP)
- **Definition**: An MDP is a mathematical model used for decision-making where outcomes are partly random and partly under the control of a decision-maker.
- **Components**:
  - **States (S)**: The set of all possible states in the environment.
  - **Actions (A)**: The set of all possible actions the agent can take.
  - **Transition Function (T)**: Probability of moving from one state to another, given an action, $ T(s_{t+1} | s_t, a_t) \).
  - **Reward Function (R)**: Immediate reward received after transitioning from one state to another due to an action, $ R(s_t, a_t) \).
  - **Initial State Distribution (P)**: Probability distribution of the initial state $ s_0 \).
  - **Discount Factor ($\gamma\))**: A factor between 0 and 1 that represents the importance of future rewards.

#### Partially Observable Markov Decision Process (POMDP)
- **Definition**: A POMDP extends MDPs to situations where the agent cannot directly observe the underlying state of the environment.
- **Components**:
  - **States (S), Actions (A), Transition Function (T), Reward Function (R), Initial State Distribution (P), Discount Factor ($\gamma\))**: Same as in MDP.
  - **Observations (O)**: The set of observations that provide partial information about the state.
  - **Observation Function (O)**: Probability of making an observation given a state, $ O(o_t | s_t) \).
- **Execution**: The agent maintains a history $ h_t = (o_0, a_0, \ldots, o_t) \) of actions and observations to infer the hidden state and make decisions.

#### Belief MDP
- **Definition**: A belief MDP reformulates a POMDP by using a belief state, which is a probability distribution over possible states, to summarize the history.
- **Belief State (b)**: Represents the probability distribution over the possible states given the history of actions and observations.
- **Belief Update**: The belief state is updated recursively based on the new observation and action taken.
  - Initial belief: $ b_0(s_0) = P(s_0)O(o_0 | s_0) \)
  - Updated belief: $ b_{t+1}(s_{t+1}) = O(o_{t+1} | s_{t+1}) \sum_{s_t} T(s_{t+1} | s_t, a_t) b_t(s_t) \)

#### Planning in POMDPs
- **Goal**: Find a policy that maximizes the expected return from any initial belief state.
- **Q-function in Belief MDP**: Defines the expected return starting from a belief state and taking an action, following the optimal policy.
  - $ Q(b, a) = \mathbb{E} \left[ R(s_t, a_t) + \gamma \max_{a'} Q(b', a') \right] \)
- **Planning Process**:
  - **Belief Update**: Update beliefs based on actions and observations.
  - **Policy Computation**: Compute policies that maximize the expected return using techniques like value iteration or policy iteration in the belief space.

#### Planning Requires POMDP Model
- **POMDP Model (P, O, A)**:
  - **P (States)**: Set of all possible states.
  - **O (Observations)**: Set of all possible observations.
  - **A (Actions)**: Set of all possible actions.

#### Model-Based Reinforcement Learning (RL)
- **Definition**: Uses a model of the environment to simulate and plan actions.
- **Components**:
  - **World Model (q_\theta)**: An approximation of the environment's dynamics, $ q_\theta(r, o' | h, a) \).
  - **Imagined Trajectories**: Simulate possible future trajectories using the model to plan and optimize policies.
- **Algorithm Example**: Dreamer algorithm, which uses a variational RNN and a distributional value function to predict future observations and rewards, optimizing the policy through imagined interactions.

### Key Points
- **MDPs** provide a framework for decision-making with full state observability.
- **POMDPs** extend MDPs to partially observable environments, where the state is not directly visible.
- **Belief MDPs** convert the POMDP problem into an MDP in the belief space, allowing the use of MDP planning techniques.
- **Planning in POMDPs** involves updating beliefs and computing policies that maximize expected returns.
- **Model-based RL** leverages models of the environment to simulate and plan, improving efficiency in POMDPs by predicting future states and rewards.

## Blond Vénère
## Introduction to Gradient-Based Direct Policy Search
Here's a summary and explanation of the key points from the provided slides on Gradient-Based Direct Policy Search:

### Markov Decision Process (MDP)
An MDP is represented by its model $ M = (S,A,T,R, p_0, \gamma) \):

- **States $ s_t \in S \)**: Represents the different situations in the environment.
- **Actions $ a_t \in A \)**: The set of all possible actions.
- **Transition distribution $ T(s_{t+1} | s_t, a_t) \)**: The probability of moving to a new state $ s_{t+1} \) given the current state $ s_t \) and action $ a_t \).
- **Reward function $ r_t = R(s_t, a_t) \)**: The immediate reward received after transitioning to a new state.
- **Initial distribution $ p_0(s_0) \)**: The distribution over initial states.
- **Discount factor $ \gamma \in [0, 1[ \)**: A factor used to discount future rewards.

### Direct Search
Direct policy search methods aim to learn a policy directly rather than solving intermediate problems like estimating value functions. This approach focuses on optimizing the true control objective directly, which can be advantageous in certain scenarios, especially when dealing with continuous state-action spaces.

### Difference Between Gradient-Based and Value-Based Methods
**Gradient-Based Methods (Direct Policy Search)**
- Directly optimize the policy by adjusting parameters to maximize the expected return.
- Use policy gradient methods where the policy is often represented as a differentiable function $ \pi_\theta \).
- More suited for continuous action spaces and can handle high-dimensional policy spaces.

**Value-Based Methods**
- Estimate the value of states (or state-action pairs) and derive a policy from these estimates.
- Methods like Q-learning fall under this category.
- Can become complex and less smooth when dealing with large or continuous state spaces.

### Why Learn Policy Directly and Not Q-Function
Learning a policy directly can often be simpler and more effective in high-dimensional or continuous action spaces. Direct policy search methods optimize the expected return directly and can sometimes discover simpler policies compared to the complex value functions needed in value-based methods.

### REINFORCE Algorithm
The REINFORCE algorithm is a Monte-Carlo policy gradient method that updates policy parameters using sampled trajectories. It is unbiased but can have high variance.

**Algorithm Steps:**
1. Initialize policy parameters $ \theta \) randomly.
2. Sample $ n \) trajectories using the current policy.
3. Update policy parameters $ \theta \) using the gradient of the expected return.

### Value Critic (Actor-Critic Methods)
Actor-Critic methods use a value function (critic) to reduce the variance of the policy gradient estimates. The critic evaluates the current policy by estimating the value function, while the actor updates the policy based on this evaluation.

### Local Optimality
Policy gradient methods can converge to local optima. To mitigate this, techniques like entropy regularization are used to maintain exploration by encouraging policies with higher entropy, thus preventing premature convergence to deterministic policies.

### Conclusions
- Direct policy search directly optimizes the policy, which can be more straightforward in certain environments.
- Policy gradient methods leverage gradient ascent on the expected return.
- Variance reduction techniques, such as Actor-Critic methods, improve the stability and efficiency of learning.
- Entropy regularization helps avoid local optima by maintaining exploration.

## Advanced Policy-Gradient Algorithms
Here's a detailed explanation and clarification of the specified concepts from the provided "Advanced Policy Gradient" slides:

### Advantage Actor-Critic (A2C)

The Advantage Actor-Critic (A2C) method combines two key components:
1. **Actor:** Updates the policy parameters to maximize the expected return.
2. **Critic:** Evaluates the policy by estimating the value function.

The updates for the actor and critic are as follows:

- **Actor Update Direction:**
  $$
  \hat{\nabla}_{\theta} J(\pi_\theta) = \left\langle \sum_{t=0}^{T-1} \gamma^t \left( \sum_{t'=t}^{T-1} \gamma^{t'-t} r_{t'} + \gamma^T V_\phi(s_T) - V_\phi(s_t) \right) \nabla_\theta \log \pi_\theta(a_t | s_t) \right\rangle_n
  $$

- **Critic Update Direction:**
  $$
  \hat{\nabla} L(\phi) = \left\langle \left( V_\phi(s_t) - \sum_{t'=t}^{T-1} \gamma^{t'-t} r_{t'} - \gamma^{T-t} V_\phi(s_T) \right) \sum_{t=0}^{T-1} \nabla_\phi V_\phi(s_t) \right\rangle_n
  $$

The actor updates the policy parameters $\theta\) based on the advantage function, which measures the benefit of a particular action over the average action in a given state. The critic updates its estimate of the value function parameters $\phi\) to better predict the expected return from each state.

### Natural Policy Gradient (NPG)

The Natural Policy Gradient approach addresses the inefficiency of standard policy gradient methods by considering the geometry of the parameter space. It uses a different norm for parameter updates, accounting for the changes in the underlying policy distribution.

- **Natural Gradient Definition:**
  $$
  \nabla_{\theta} J(\theta) = F(\theta)^{-1} \nabla_{\theta} J(\theta)
  $$
  where $F(\theta)\) is the Fisher Information Matrix:
  $$
  F(\theta) = \mathbb{E}_{s \sim d_{\pi_\theta}, a \sim \pi_\theta} \left[ (\nabla_\theta \log \pi_\theta(a | s)) (\nabla_\theta \log \pi_\theta(a | s))^T \right]
  $$

Natural gradients account for small variations in the policy distribution, making the updates more stable. However, computing $F(\theta)^{-1}\) can be computationally expensive. Practical implementations often use approximations like the conjugate gradient method or solve a least-squares minimization problem.


- **Advantage Actor-Critic:**
  - Combines actor and critic to reduce variance and improve learning efficiency.
  - Actor updates policy parameters using advantage estimates.
  - Critic updates value function estimates to provide accurate advantage calculations.

- **Natural Policy Gradient:**
  - Adjusts standard policy gradient methods by considering the geometry of the parameter space.
  - Uses the Fisher Information Matrix to measure the change in the policy distribution.
  - More stable updates compared to standard policy gradients but computationally expensive.

## Pascal Leroy
## Multi-Agent Reinforcement Learning


### MDP
A Markov Decision Process (MDP) in single-agent reinforcement learning is defined by:
- **States $ s \in S \)**: Different situations the agent can be in.
- **Actions $ u \in U \)**: Possible actions the agent can take.
- **Transition Function** $ P(s_{t+1} | s_t, u_t) \): Probability of moving to a new state $ s_{t+1} \) given the current state $ s_t \) and action $ u_t \).
- **Reward Function** $ r_t = R(s_{t+1}, s_t, u_t) \): Reward received after transitioning to a new state.
- **Policy** $ \pi(u_t | s_t) \): Probability distribution over actions given the current state.
- **Goal**: Maximize the total expected sum of discounted rewards $ \sum_{t=0}^{T} \gamma^t r_t \) with a discount factor $ \gamma \in [0, 1) \) using the optimal policy $ \pi^* \).

### DQN
Deep Q-Network (DQN) is a Q-learning algorithm that uses a neural network parameterized by $ \theta \) to approximate the Q-value function.
- **Loss Function**:
  $$
  L(\theta) = \mathbb{E}_{\langle s_t, u_t, r_t, s_{t+1} \rangle \sim B} \left[ \left( r_t + \gamma \max_{u \in U} Q(s_{t+1}, u; \theta') - Q(s_t, u_t; \theta) \right)^2 \right]
  $$
- **Replay Buffer $ B \)**: A collection of transitions used to sample and update the network.
- **Target Network $ \theta' \)**: A copy of $ \theta \) that is periodically updated.

### Actor-Critic
The Advantage Actor-Critic (A2C) method combines two components:
1. **Actor**: Learns the policy $ \pi \).
   $$
   \nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t} A(s_t, u_t; \phi) \nabla_\theta \log \pi_\theta(u_t | s_t) \right]
   $$
2. **Critic**: Evaluates the policy by estimating the advantage function $ A(s_t, u_t; \phi) = Q(s_t, u_t) - V(s_t) \).

### Framework MARL Definitions
In Multi-Agent Reinforcement Learning (MARL), the framework is defined by a Stochastic Game (also known as a Markov Game):
- **Agents** $ \{a_1, \ldots, a_n\} \).
- **States** $ s \in S \).
- **Actions** $ U = U_1 \times \ldots \times U_n \).
- **Transition Function** $ P(s_{t+1} | s_t, u_t) \).
- **Reward Function** $ r_i^t = R_i(s_{t+1}, s_t, u_t) \) for each agent.
- **Observation Function** $ O: S \times \{1, \ldots, n\} \to Z \).
- **Goal**: Maximize the total expected sum of discounted rewards for each agent.

### Different Types of Settings
1. **Cooperative Setting**: All agents share a common goal.
   - **Example**: Traffic control, robotics teams.
   - **Problem**: Credit assignment and non-stationarity.
   - **Solution**: Centralized Training with Decentralized Execution (CTDE), QMIX.
   
2. **Competitive Setting**: The gain of one agent equals the loss of another (zero-sum).
   - **Example**: Board games, video games.
   - **Problem**: Adversarial policies.
   - **Solution**: Minimax-Q, AlphaGo Zero.

3. **General Sum Setting**: Mix of cooperation and competition.
   - **Example**: Team-based video games, autonomous vehicles.

### QMIX
QMIX uses a monotonic value function factorization to ensure individual global maximization (IGM) in cooperative settings.
- **Architecture**:
  - A hypernetwork computes the weights of a second neural network.
  - Ensures monotonicity by constraining the weights to be positive.

### DQV and DQV Mix
- **DQV**: Simultaneously learns Q-values and value functions to reduce overestimation in DQN.
  $$
  L(\theta) = \mathbb{E} \left[ \left( r_t + \gamma V(s_{t+1}; \phi') - Q(s_t, u_t; \theta) \right)^2 \right]
  $$
  $$
  L(\phi) = \mathbb{E} \left[ \left( r_t + \gamma V(s_{t+1}; \phi') - V(s_t; \phi) \right)^2 \right]
  $$
- **QVMix**: Extends DQV to cooperative settings using QMIX principles.

### Infrastructure Management
CTDE methods can be applied to infrastructure management for inspecting and repairing systems over time to reduce costs and risks.
- **State**: Belief on the deterioration state.
- **Actions**: Do nothing, inspect, repair/replace.
- **Reward**: Includes costs related to failure, inspection, and repairs.

### Communication in MARL
- **Advantages**:
  - Improved coordination among agents.
  - Enhanced learning efficiency.
- **Disadvantages**:
  - Communication overhead.
  - Potential for non-stationarity.
- **Examples**:
  - **RIAL**: Reinforced Inter-Agent Learning.
  - **DIAL**: Differentiable Inter-Agent Learning with CTDE.

### Competitive Settings
- **AlphaGo Zero**: Uses self-play and MCTS with a neural network to master Go.
- **Hide and Seek**: Emergent behaviors in a simulated environment where hiders avoid seekers.

### Adversarial Attacks
Adversarial policies can disrupt the learning process by exploiting vulnerabilities in the policies of other agents.
- **Types**:
  - **Observation Attacks**: Manipulate the observations received by agents.
  - **Action Attacks**: Influence the actions taken by agents.
- **Solutions**:
  - **Robust Training**: Train agents against a variety of adversarial policies.
  - **Fine-Tuning**: Adjust policies to defend against known attacks.

## Arthur Louette 
## Robotic reinforcement learning
### Introduction: Why does it change with robotics in the real world?

**Robotic reinforcement learning** involves controlling robots in complex, real-world environments, which brings unique challenges such as:
- **Safety**: Random exploration could damage the robot.
- **Sample efficiency**: Real-time operations limit training time.
- **Real-time decision**: Quick processing and execution are necessary.
- **Generalization**: Variability in real-world conditions like lighting and textures.
- **Sensor noise and uncertainty**: Inaccurate sensors add complexity.

### Simulation: What is a simulator?

A **robotic simulator** uses a physics engine to replicate real-world environments. It enables:
- Safe learning.
- Increased data acquisition via parallelization.
- Realistic training scenarios.
- Easy environment reset.

### Pas retenir les details (Don't retain the details)

Focus on understanding the main concepts rather than memorizing every detail.

### Reward Hacking + EUREKA

**Reward Hacking**: When an agent exploits unintended actions to maximize rewards, often leading to undesired behaviors. Address it by:
- Fixing environment bugs.
- Careful reward design.
- Using exploration strategies.

**EUREKA**: Uses Large Language Models (LLMs) to design reward functions from high-level descriptions, improving learning outcomes.

### How to avoid reward sparsity -> better dense + problems

Sparse rewards make convergence difficult, while dense rewards facilitate incremental learning but are hard to design. Solutions include:
- Proper reward shaping.
- Combining multiple objectives.
- Using advanced knowledge of the environment.

### Euraka repasser dessus pas super important grandes lignes (EUREKA overview, not super important, main points)

EUREKA uses LLMs to generate reward functions, improving task performance and efficiency.

### Sim to real gap problem (very important)

**Sim2Real Gap**: Disparity between performance in simulation and reality due to model inaccuracies and sensor differences. Solutions include:
- **System Identification**: Directly identify and simulate real-world states.
- **Domain Randomization**: Increase domain variability to include real environments.
- **Domain Adaptation**: Adapt simulations to better match real environments.

### Techniques to avoid problems and their benefits

**Direct System Identification**: Directly simulate identified real-world parameters but challenging to be precise.
**Domain Randomization**: Broadens domain variability, simple but less sample efficient.
**Domain Adaptation**: More efficient and better performance, needs real data.

### Randomizations and adaptations

- **Domain Randomization**: Increases sample efficiency by simulating various scenarios.
- **Domain Adaptation**: Tailors simulations to better match real-world conditions, enhancing transferability.

### Apprentissage reel (real learning) advantages

**Real-World Learning**: 
- Directly overfits to real-world conditions.
- Emergence of interesting patterns.
- Improved sample efficiency and use of prior data.

### How to improve the algorithm

**Improvements**:
- Increase sample efficiency with better algorithms (e.g., SAC, TD3).
- Use extensive gradient steps with good regularization.
- Higher update-to-data ratio.

### Sample efficiency, gradient to regularization

- **Sample Efficiency**: Use off-policy algorithms and replay buffers.
- **Regularization**: Prevents overfitting, crucial for effective learning.

### Update to data ratio

Higher update-to-data ratio enhances learning efficiency and model performance.

### Explain exploration and problems in real-world + how to avoid

- **Exploration**: Adding noise to actions for better exploration.
- **Problems**: High-frequency noise can damage robots. Use state-dependent exploration to mitigate risks.
- **Avoiding Problems**: Smooth and constrained exploration strategies.

### Paper less important

Some sections or papers may be less critical for understanding the core concepts.

### FastRLAP: Demo technique for real use

**FastRLAP**: Uses sample-efficient reinforcement learning and pretraining for high-speed driving tasks. Demonstrates quick learning and practical applications.

### Examples (final)

Provides practical examples and demonstrations of techniques discussed.

### What is a foundation model?

A **foundation model** is a broad-trained machine learning model applicable across various tasks, with superior generalization capabilities.

### How to create a multi-task RL model + example (last slides)

**Multi-Task RL Model**:
- **Approach**: Leverage prior experiences and data from similar tasks.
- **Example**: Q-Transformer uses large-scale data and language instructions for scalable learning across multiple tasks.

## Lize Pirenne 
## Reinforcement learning and Large Language Models
#### **Outline**
1. Large Language Models (LLMs)
2. Reinforcement Learning (RL)
3. Learning from humans
4. RL Methods for LLMs
5. Current Challenges


#### **Large Language Models**

- **Language Models:**
  - Estimate the probability of the next word in a sentence.
  - Use causal language modeling to predict the next word.

- **Architecture:**
  - Decoder-only transformer architecture is common for generative models.
  - Operate on tokens using methods like byte-pair encoding (BPE).

- **Generation:**
  - Produces a distribution over the vocabulary for each token.
  - Uses strategies like greedy decoding, beam search, and sampling methods (temperature, nucleus, top-k).

- **Applications:**
  - Can handle sequential data, including text, code, images, and sounds.
  - Multi-modal generation involves specific encoders and pre-trained models.


#### **Reinforcement Learning**

- **Fundamentals:**
  - An agent interacts with an environment by taking actions to maximize rewards.
  - Value-based methods (Q-learning) and policy-based methods (gradient-based optimization) are common approaches.

- **Choice of Approach:**
  - Policy-based methods are preferred for large action spaces due to efficiency and stability.


#### **Learning from Humans**

- **Imitation Learning:**
  - Policy is learned from a set of demonstrations.
  - Behavioral cloning mimics expert actions but can be brittle and biased.

- **RL in LLMs:**
  - Next token prediction parallels behavioral cloning.
  - Problems like open-loop drifting and hallucinations occur in LLMs.


#### **RL Methods for LLMs**

- **Inverse Reinforcement Learning (IRL):**
  - Learns a reward function from demonstrations.
  - Addresses reward ambiguity and requires environment access.

- **Human Feedback:**
  - Feedback can be in the form of preferences, rankings, scores, and corrections.
  - The policy gradient theorem is used to update the policy based on expected returns.

- **REINFORCE Algorithm:**
  - Updates the policy using the policy gradient theorem.
  - Variance reduction is achieved using a baseline.


#### **Reinforcement Learning from Human Feedback (RLHF)**

- **Challenges:**
  - Reward hacking where models exploit the reward function without solving the task.
  - Enhanced RLHF includes methods to mitigate reward hacking.

- **Examples of RLHF:**
  - Misaligned supervised fine-tuning data can lead to hallucinations.
  - Human preference data is used to train reward models.
  - Factually augmented RLHF and iterative training improve performance.

- **Direct Preference Optimization (DPO):**
  - Uses LLMs as their own reward models.
  - Regularization terms can be added to improve performance.


#### **Current Challenges**

- **Task Complexity:**
  - Evaluating the complexity of tasks and selecting appropriate models.
  - Managing tool usage and multi-step planning for specific goals.

- **Future Directions:**
  - Continuous learning, personalization, efficient data usage, and evaluation methods are ongoing challenges.


#### **Take-home Message**

- **Use LLMs for what they excel at, and continuously strive for improvements through reinforcement learning and human feedback.**
