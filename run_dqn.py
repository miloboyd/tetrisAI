# run_dqn.py

import os
import numpy as np
import pygame
from pygame import time
import torch
import torch.nn as nn
import torch.optim as optim
from tetris_env import TetrisEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH  = 'dqn_checkpoint.pth'
REWARDS_CSV      = 'training_rewards.csv'

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        return map(np.array, zip(*batch))

    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        flat_dim = np.prod(obs_shape)
        self.model = nn.Sequential(
            nn.Linear(flat_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def forward(self, x):
        x = x.to(device).float()
        x = x.flatten(start_dim=1)
        return self.model(x)

def run():
    env = TetrisEnv()
    env.pyRender(True)
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n

    model = DQN(obs_shape, n_actions).to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)
    try:
        model.load_state_dict(checkpoint['model_state'])
    except:
        try:
            model.load_state_dict(checkpoint) # when using 'normally generated' pth as opposed to above line
        except:
            pass
    model.eval()

    def select_action(state):
        epsilon_eval = 0.00
        if np.random.rand() < epsilon_eval:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                q_vals = model(state_tensor)
                action = int(q_vals.argmax())
        return action

    # Run a few test episodes
    num_test_episodes = 5

    for episode in range(num_test_episodes):
        env.game.game_over = False
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            if env.render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done = True
                        break

            action = select_action(state)
            #print(f"Action {action:.2f}")
            state, reward, done, _ = env.step(action)
            total_reward += reward

            # Wait for x milliseconds to better interpret AI actions
            time.delay(100)
        print(f"Total Reward: {total_reward:.2f}")

    env.close()


if __name__ == '__main__':
    run()