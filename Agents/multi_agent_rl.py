# agents/multi_agent_rl.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from env.icu_env import ICUMultiEnv

# -----------------------------
# Simple Policy Network
# -----------------------------
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=-1)
        return x

# -----------------------------
# REINFORCE Agent
# -----------------------------
class REINFORCEAgent:
    def __init__(self, env: ICUMultiEnv, lr=1e-2, gamma=0.99):
        self.env = env
        self.gamma = gamma
        self.log_probs = []
        self.rewards = []

        # Start with max 5 patients for simplicity
        self.input_dim = 5
        self.action_dim = 5
        self.policy = PolicyNetwork(self.input_dim, self.action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def select_action(self, state):
        # Pad state to fixed length
        state = torch.tensor(state + [0]*(self.input_dim-len(state)), dtype=torch.float32)
        probs = self.policy(state)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        self.log_probs.append(m.log_prob(action))
        return action.item()

    def finish_episode(self):
        R = 0
        returns = []
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = []
        for log_prob, R in zip(self.log_probs, returns):
            loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        torch.stack(loss).sum().backward()
        self.optimizer.step()

        # Clear memory
        self.log_probs = []
        self.rewards = []

# -----------------------------
# Training Loop
# -----------------------------
if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Initialize environment with 2 agents
    env = ICUMultiEnv(num_agents=2)
    agents = [REINFORCEAgent(env) for _ in range(env.num_agents)]

    # Logging lists
    states_log = []
    actions_log = []

    episodes = 50
    for ep in range(episodes):
        env.reset()
        # Add example patients
        env.add_patient(severity=2.5)
        env.add_patient(severity=3.2)
        env.add_patient(severity=1.8)

        done = False
        while not done:
            state = [p.severity for p in env.get_pending_patients()]
            if not state:
                break

            # Agents select actions
            actions = [agent.select_action(state) for agent in agents]

            # Step environment
            next_state, total_reward, done = env.step_multi(actions)

            # Store shared reward for all agents
            for i, agent in enumerate(agents):
                agent.rewards.append(total_reward)

            # Log states and actions
            states_log.append(state.copy())
            actions_log.append(actions.copy())

        # Update all agents after episode
        for agent in agents:
            agent.finish_episode()

        print(f"Episode {ep+1} finished")

    # Save models
    for i, agent in enumerate(agents):
        torch.save(agent.policy.state_dict(), f"models/multi_agent_rl_{i}.pt")
    print("Multi-agent models saved to models/")

    # Save logs for surrogate tree / XAI
    np.save("logs/multi_states.npy", np.array(states_log, dtype=object))
    np.save("logs/multi_actions.npy", np.array(actions_log, dtype=object))
    print(f"Saved {len(states_log)} multi-agent state-action pairs to logs/")
