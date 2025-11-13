# agents/single_agent_rl.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from env.icu_env import ICUEnv
import os

# -------------------------
# Simple Policy Network
# -------------------------
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=-1)
        return x

# -------------------------
# REINFORCE Agent
# -------------------------
class REINFORCEAgent:
    def __init__(self, env: ICUEnv, lr=1e-2, gamma=0.99):
        self.env = env
        self.gamma = gamma
        self.log_probs = []
        self.rewards = []

        self.input_dim = 5     # max number of patients in state
        self.action_dim = 5    # max number of possible actions
        self.policy = PolicyNetwork(self.input_dim, self.action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def select_action(self, state):
        # pad state to input_dim
        state_tensor = torch.tensor(state + [0]*(self.input_dim - len(state)), dtype=torch.float32)
        probs = self.policy(state_tensor)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        self.log_probs.append(m.log_prob(action))
        # Clamp action to valid range
        action = min(action.item(), len(state)-1)
        return action

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

        self.log_probs = []
        self.rewards = []

    def train(self, episodes=50):
        for ep in range(episodes):
            self.env.reset()
            # Add example patients
            self.env.add_patient(severity=2.5)
            self.env.add_patient(severity=3.2)
            self.env.add_patient(severity=1.8)

            done = False
            while not done:
                state = [p.severity for p in self.env.get_pending_patients()]
                if not state:
                    break
                action = self.select_action(state)
                next_state, reward, done = self.env.step_rl(action)
                self.rewards.append(reward)
            self.finish_episode()
            print(f"Episode {ep+1} finished")

# -------------------------
# Main training + logging
# -------------------------
if __name__ == "__main__":
    env = ICUEnv()
    agent = REINFORCEAgent(env)

    # Train RL agent
    agent.train(episodes=50)
    os.makedirs("models", exist_ok=True)
    torch.save(agent.policy.state_dict(), "models/single_agent_rl.pt")
    print("Model saved to models/single_agent_rl.pt")

    # Logging state-action pairs for surrogate tree
    states = []
    actions = []

    episodes = 10  # small number for logging
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
            action = agent.select_action(state)
            next_state, reward, done = env.step_rl(action)

            # log state-action
            states.append(state.copy())
            actions.append(action)

    # Save logs
    os.makedirs("logs", exist_ok=True)
    np.save("logs/states.npy", np.array(states, dtype=object))
    np.save("logs/actions.npy", np.array(actions))
    print(f"Saved {len(states)} state-action pairs to logs/states.npy & logs/actions.npy")
