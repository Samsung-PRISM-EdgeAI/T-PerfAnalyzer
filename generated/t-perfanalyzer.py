import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple

# Define a named tuple for storing experience tuples
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'log_prob'))

class TPerfAnalyzerSimulator:
    """
    A simulator for T-PerfAnalyzer, providing feedback on generated test cases.

    This class simulates the real-world performance monitoring and energy profiling
    that T-PerfAnalyzer would perform on a TV application. It takes a generated
    test case (as a string) and a current application state, and returns
    simulated performance metrics and energy consumption. These metrics are then
    used to calculate a reward for the PPO agent.

    The simulation is simplified for demonstration purposes, assuming that
    certain keywords or patterns in the test case, combined with the state,
    lead to specific performance outcomes.

    Attributes
    ----------
    performance_goals : dict
        Target performance metrics (e.g., {'cpu_usage': 0.1, 'memory_usage': 0.2}).
    energy_budget : float
        Maximum allowed energy consumption.
    bug_detection_rate : float
        Probability of detecting a 'bug' keyword in a test case.
    """
    def __init__(self, performance_goals=None, energy_budget=10.0, bug_detection_rate=0.3):
        self.performance_goals = performance_goals if performance_goals is not None else {
            'cpu_usage': 0.15, 'memory_usage': 0.25, 'latency_ms': 50
        }
        self.energy_budget = energy_budget
        self.bug_detection_rate = bug_detection_rate

    def _simulate_metrics(self, test_case: str, state: np.ndarray) -> dict:
        """
        Simulates performance and energy metrics based on the test case and state.

        Parameters
        ----------
        test_case : str
            The generated test case string.
        state : np.ndarray
            The current state vector of the TV application.

        Returns
        -------
        dict
            A dictionary containing simulated metrics like 'cpu_usage',
            'memory_usage', 'latency_ms', 'energy_consumption', 'bugs_found'.
        """
        # Simple simulation logic:
        # - Test cases with 'stress' or 'heavy' might increase resource usage.
        # - Test cases with 'optimize' or 'efficient' might decrease resource usage.
        # - State features can also influence outcomes.
        base_cpu = 0.2 + state[0] * 0.1 # state[0] could be prev_cpu_usage
        base_mem = 0.3 + state[1] * 0.1 # state[1] could be prev_mem_usage
        base_latency = 100 + state[2] * 20 # state[2] could be prev_latency

        if "stress" in test_case.lower() or "heavy" in test_case.lower():
            base_cpu *= 1.5
            base_mem *= 1.3
            base_latency *= 1.2
        elif "optimize" in test_case.lower() or "efficient" in test_case.lower():
            base_cpu *= 0.8
            base_mem *= 0.9
            base_latency *= 0.9
        
        # Add some randomness
        cpu_usage = max(0.05, min(1.0, base_cpu + random.uniform(-0.05, 0.05)))
        memory_usage = max(0.1, min(1.0, base_mem + random.uniform(-0.05, 0.05)))
        latency_ms = max(20, base_latency + random.uniform(-10, 10))
        energy_consumption = (cpu_usage * 5 + memory_usage * 3 + latency_ms * 0.02) * random.uniform(0.8, 1.2)

        bugs_found = 0
        if "bug" in test_case.lower() or "error" in test_case.lower():
            if random.random() < self.bug_detection_rate:
                bugs_found = 1 # Simulate finding a bug

        return {
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'latency_ms': latency_ms,
            'energy_consumption': energy_consumption,
            'bugs_found': bugs_found
        }

    def evaluate_test_case(self, test_case: str, state: np.ndarray) -> tuple[dict, float]:
        """
        Evaluates a generated test case and returns metrics and a reward.

        Parameters
        ----------
        test_case : str
            The generated test case string.
        state : np.ndarray
            The current state vector of the TV application.

        Returns
        -------
        tuple[dict, float]
            A tuple containing:
            - metrics (dict): Simulated performance and energy metrics.
            - reward (float): The calculated reward based on metrics and goals.
        """
        metrics = self._simulate_metrics(test_case, state)

        # Calculate reward based on performance goals and energy budget
        reward = 0.0

        # Performance reward: Penalize deviation from goals
        reward -= abs(metrics['cpu_usage'] - self.performance_goals['cpu_usage']) * 5.0
        reward -= abs(metrics['memory_usage'] - self.performance_goals['memory_usage']) * 5.0
        reward -= abs(metrics['latency_ms'] - self.performance_goals['latency_ms']) / 100.0

        # Energy efficiency reward: Penalize exceeding budget, reward staying below
        if metrics['energy_consumption'] > self.energy_budget:
            reward -= (metrics['energy_consumption'] - self.energy_budget) * 0.5
        else:
            reward += (self.energy_budget - metrics['energy_consumption']) * 0.1 # Small reward for efficiency

        # Bug detection reward: Significant positive reward for finding bugs
        if metrics['bugs_found'] > 0:
            reward += 10.0 * metrics['bugs_found']

        # Reward for generating diverse or effective test cases (simplified)
        if "edge case" in test_case.lower() or "boundary" in test_case.lower():
            reward += 1.0

        return metrics, reward

class LLMEmulator:
    """
    Emulates the behavior of a Large Language Model for test case generation.

    This class simulates an LLM's ability to take a prompt template and
    current application state, and generate a concrete test case string.
    In a real scenario, this would involve API calls to a powerful LLM.
    Here, it uses simple string formatting and keyword-based logic.

    Attributes
    ----------
    prompt_templates : list of str
        A list of predefined prompt templates the LLM can use.
    """
    def __init__(self, prompt_templates: list[str]):
        self.prompt_templates = prompt_templates

    def generate_test_case(self, prompt_idx: int, state: np.ndarray) -> str:
        """
        Generates a test case string using a selected prompt template and state.

        Parameters
        ----------
        prompt_idx : int
            The index of the prompt template to use from `self.prompt_templates`.
        state : np.ndarray
            The current state vector of the TV application.

        Returns
        -------
        str
            The generated test case string.
        """
        if not (0 <= prompt_idx < len(self.prompt_templates)):
            raise ValueError(f"Prompt index {prompt_idx} out of bounds for {len(self.prompt_templates)} templates.")

        template = self.prompt_templates[prompt_idx]

        # Simple state-based parameterization for the LLM emulator
        # In a real LLM, the state would be part of the context or system prompt.
        state_info = {
            "prev_cpu": f"{state[0]*100:.1f}%",
            "prev_mem": f"{state[1]*100:.1f}%",
            "prev_latency": f"{state[2]:.0f}ms",
            "prev_energy": f"{state[3]:.1f}J",
            "prev_bugs": f"{int(state[4])}",
            "focus_area": ["UI Responsiveness", "Video Playback", "App Launch", "Network Stability"][int(state[5])]
        }

        # Format the template with state information
        try:
            generated_text = template.format(**state_info)
        except KeyError:
            # Fallback if template doesn't use all state_info keys
            generated_text = template + f" (Context: CPU {state_info['prev_cpu']}, Focus: {state_info['focus_area']})"

        # Add some "LLM-like" embellishment based on keywords
        if "stress" in template.lower():
            generated_text += " - Execute with maximum concurrent users and high-resolution content."
        elif "optimize" in template.lower():
            generated_text += " - Focus on identifying bottlenecks and proposing efficiency improvements."
        elif "bug" in template.lower():
            generated_text += " - Specifically look for crashes or unexpected behavior."
        
        return generated_text

class ActorNetwork(nn.Module):
    """
    Policy network (Actor) for PPO, mapping states to action probabilities.

    This network takes the current state of the TV application as input
    and outputs a probability distribution over the available prompt templates.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the actor network.

        Parameters
        ----------
        state : torch.Tensor
            The input state tensor.

        Returns
        -------
        torch.Tensor
            Log probabilities of actions.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # Output logits, which will be converted to probabilities by a Categorical distribution
        return self.fc3(x)

class CriticNetwork(nn.Module):
    """
    Value network (Critic) for PPO, estimating the value of a given state.

    This network takes the current state of the TV application as input
    and outputs a single scalar value, representing the estimated
    expected cumulative reward from that state.
    """
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1) # Output a single value

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the critic network.

        Parameters
        ----------
        state : torch.Tensor
            The input state tensor.

        Returns
        -------
        torch.Tensor
            The estimated value of the state.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class PPOAgent:
    """
    Proximal Policy Optimization (PPO) agent for adaptive prompt selection.

    This agent learns to select the most effective prompt templates for
    generating test cases by interacting with the T-PerfAnalyzerSimulator
    and optimizing its policy using the PPO algorithm.

    Attributes
    ----------
    state_dim : int
        Dimension of the state space.
    action_dim : int
        Dimension of the action space (number of prompt templates).
    gamma : float
        Discount factor for future rewards.
    gae_lambda : float
        Lambda parameter for Generalized Advantage Estimation (GAE).
    ppo_epsilon : float
        Clipping parameter for the PPO loss function.
    ppo_epochs : int
        Number of optimization epochs per policy update.
    batch_size : int
        Mini-batch size for PPO updates.
    lr_actor : float
        Learning rate for the actor (policy) network.
    lr_critic : float
        Learning rate for the critic (value) network.
    entropy_coeff : float
        Coefficient for the entropy bonus in the policy loss.
    value_loss_coeff : float
        Coefficient for the value loss in the total loss.
    device : torch.device
        The device (CPU or GPU) to run computations on.
    actor : ActorNetwork
        The policy network.
    critic : CriticNetwork
        The value network.
    actor_optimizer : torch.optim.Adam
        Optimizer for the actor network.
    critic_optimizer : torch.optim.Adam
        Optimizer for the critic network.
    rollout_buffer : list of Experience
        Buffer to store collected experiences for PPO updates.
    """
    def __init__(self, state_dim: int, action_dim: int,
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 ppo_epsilon: float = 0.2, ppo_epochs: int = 10,
                 batch_size: int = 64, lr_actor: float = 3e-4,
                 lr_critic: float = 1e-3, entropy_coeff: float = 0.01,
                 value_loss_coeff: float = 0.5, hidden_dim: int = 128):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ppo_epsilon = ppo_epsilon
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.entropy_coeff = entropy_coeff
        self.value_loss_coeff = value_loss_coeff

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = CriticNetwork(state_dim, hidden_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.rollout_buffer = []

    def select_action(self, state: np.ndarray) -> tuple[int, float]:
        """
        Selects an action (prompt index) based on the current policy.

        Parameters
        ----------
        state : np.ndarray
            The current state vector.

        Returns
        -------
        tuple[int, float]
            A tuple containing:
            - action (int): The index of the selected prompt template.
            - log_prob (float): The log probability of the selected action.
        """
        # Convert state to tensor and move to device
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get action logits from actor network
        with torch.no_grad():
            action_logits = self.actor(state_tensor)
        
        # Create a categorical distribution from logits
        dist = torch.distributions.Categorical(logits=action_logits)
        
        # Sample an action
        action = dist.sample()
        
        # Get the log probability of the sampled action
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item()

    def store_experience(self, state: np.ndarray, action: int, reward: float,
                         next_state: np.ndarray, log_prob: float):
        """
        Stores an experience tuple in the rollout buffer.

        Parameters
        ----------
        state : np.ndarray
            The state before the action.
        action : int
            The action taken (prompt index).
        reward : float
            The reward received after taking the action.
        next_state : np.ndarray
            The state after taking the action.
        log_prob : float
            The log probability of the action taken.
        """
        self.rollout_buffer.append(Experience(state, action, reward, next_state, log_prob))

    def _compute_advantages_and_returns(self, rewards: list[float], values: list[float],
                                        next_value: float) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes Generalized Advantage Estimation (GAE) and discounted returns.

        Parameters
        ----------
        rewards : list of float
            List of rewards collected in a trajectory.
        values : list of float
            List of state values predicted by the critic for each state in the trajectory.
        next_value : float
            The value of the next state (or 0 if terminal).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A tuple containing:
            - advantages (torch.Tensor): Computed GAE advantages.
            - returns (torch.Tensor): Computed discounted returns.
        """
        returns = []
        advantages = []
        
        # Convert to tensors
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        values_tensor = torch.FloatTensor(values).to(self.device)
        
        # Add the next_value to the end of the values for TD error calculation
        # This is for the last step's advantage calculation
        all_values = torch.cat([values_tensor, torch.FloatTensor([next_value]).to(self.device)])

        gae = 0
        for t in reversed(range(len(rewards))):
            # TD error: delta_t = R_t + gamma * V(S_{t+1}) - V(S_t)
            delta = rewards_tensor[t] + self.gamma * all_values[t+1] - all_values[t]
            # GAE: A_t = delta_t + gamma * lambda * A_{t+1}
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae) # Insert at beginning to maintain order

            # Discounted return: G_t = R_t + gamma * G_{t+1}
            # For PPO, we use GAE for advantages and V(S_t) + A_t for returns
            # Or, simply discounted sum of rewards from that point
            # Here, we use V(S_t) + A_t as the target for the value function
            returns.insert(0, gae + all_values[t])

        return torch.FloatTensor(advantages).to(self.device), torch.FloatTensor(returns).to(self.device)

    def learn(self):
        """
        Performs a PPO update using the collected experiences in the rollout buffer.

        This method calculates advantages and returns, then iterates for multiple
        epochs over mini-batches of the collected data to update the actor and
        critic networks.
        """
        if not self.rollout_buffer:
            return

        # Extract data from buffer
        states = torch.FloatTensor(np.array([e.state for e in self.rollout_buffer])).to(self.device)
        actions = torch.LongTensor(np.array([e.action for e in self.rollout_buffer])).to(self.device)
        rewards = [e.reward for e in self.rollout_buffer]
        old_log_probs = torch.FloatTensor(np.array([e.log_prob for e in self.rollout_buffer])).to(self.device)
        
        # Get values for all states in the buffer
        with torch.no_grad():
            values = self.critic(states).squeeze().cpu().numpy().tolist()
            # The value of the last state in the buffer (next_state of the last experience)
            # If the episode ended, next_value is 0. Otherwise, it's the critic's prediction.
            last_state_tensor = torch.FloatTensor(self.rollout_buffer[-1].next_state).unsqueeze(0).to(self.device)
            next_value = self.critic(last_state_tensor).item() if self.rollout_buffer[-1].next_state is not None else 0.0

        # Compute advantages and returns
        advantages, returns = self._compute_advantages_and_returns(rewards, values, next_value)
        
        # Normalize advantages for stable training
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO optimization loop
        for _ in range(self.ppo_epochs):
            # Create mini-batches
            indices = torch.randperm(len(self.rollout_buffer))
            for start_idx in range(0, len(self.rollout_buffer), self.batch_size):
                end_idx = start_idx + self.batch_size
                batch_indices = indices[start_idx:end_idx]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # --- Update Critic (Value Network) ---
                # Predict new values for the batch states
                current_values = self.critic(batch_states).squeeze()
                # Value loss: Mean Squared Error between predicted values and target returns
                critic_loss = F.mse_loss(current_values, batch_returns)
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                # --- Update Actor (Policy Network) ---
                # Get new action probabilities and log probabilities
                action_logits = self.actor(batch_states)
                dist = torch.distributions.Categorical(logits=action_logits)
                new_log_probs = dist.log_prob(batch_actions)
                
                # Calculate ratio: pi_new(a|s) / pi_old(a|s)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # PPO clipping objective
                # PPO_loss = -min(ratio * advantage, clamp(ratio, 1-epsilon, 1+epsilon) * advantage)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.ppo_epsilon, 1.0 + self.ppo_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Entropy bonus for exploration
                entropy = dist.entropy().mean()
                
                # Total actor loss
                actor_loss = actor_loss - self.entropy_coeff * entropy

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
        
        # Clear the rollout buffer after update
        self.rollout_buffer.clear()


def T_PerfAnalyzer(
    initial_state: np.ndarray,
    prompt_templates: list[str],
    num_episodes: int = 100,
    steps_per_episode: int = 20,
    ppo_params: dict = None,
    simulator_params: dict = None,
    llm_emulator_params: dict = None
) -> tuple[list[str], list[float], list[dict]]:
    """
    The core PPO-guided agentic pipeline for adaptive prompt selection and test case generation.

    This function orchestrates the interaction between a PPO agent, an LLM emulator,
    and a T-PerfAnalyzer simulator to adaptively generate test cases for TV applications.
    It aims to improve real-time performance monitoring and energy-efficient profiling.

    The pipeline works as follows:
    1. An initial state of the TV application is provided.
    2. A PPO agent, trained to select optimal prompt templates, chooses