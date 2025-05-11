# run_dqn.py

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tetris_env import TetrisEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH  = 'dqn_checkpoint.pth'
REWARDS_CSV      = 'training_rewards.csv'

# class ReplayBuffer:
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.buffer = []
#         self.pos = 0

#     def push(self, transition):
#         if len(self.buffer) < self.capacity:
#             self.buffer.append(transition)
#         else:
#             self.buffer[self.pos] = transition
#         self.pos = (self.pos + 1) % self.capacity

#     def sample(self, batch_size):
#         indices = np.random.choice(len(self.buffer), batch_size, replace=False)
#         batch = [self.buffer[i] for i in indices]
#         return map(np.array, zip(*batch))

#     def __len__(self):
#         return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(obs_shape[0] * obs_shape[1], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        x = x.to(device).float()
        x = self.flatten(x)
        return self.net(x)

def run():
    env = TetrisEnv()
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n

    model = DQN(obs_shape, n_actions).to(device)
    model.load_state_dict(torch.load("dqn_checkpoint.pth"))
    model.eval()

    def select_action(state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = model(state_tensor)
            return q_values.argmax().item()

    # Run a few test episodes
    num_test_episodes = 1
    max_steps = 1000

    for episode in range(num_test_episodes):
        state, info = env.reset()
        total_reward = 0

        for step in range(max_steps):
            action = select_action(state)
            state, reward, done, _, _ = env.step(action)
            if done:
                break

    env.close()


if __name__ == '__main__':
    run()