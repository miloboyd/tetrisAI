# train_dqn.py

import os
import numpy as np
from pygame import time
import torch
import torch.nn as nn
import torch.optim as optim
from tetris_env import TetrisEnv

# ──────────────────────────────────────────────────────────────
# Hyperparameters (tweak as needed)
LR               = 1e-3
GAMMA            = 0.99
EPS_START        = 1.0
EPS_END          = 0.1
EPS_DECAY        = 5000
BATCH_SIZE       = 64
BUFFER_CAPACITY  = 100000
TARGET_UPDATE    = 10       # in episodes
CHECKPOINT_PATH  = 'dqn_checkpoint.pth'
REWARDS_CSV      = 'training_rewards.csv'

MAX_EPISODES     = 1000     # number of episodes per run
RESUME_TRAINING  = False    # if False, starts fresh and clears CSV
# ──────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def load_checkpoint(obs_shape, n_actions):
    """Load model, optimizer, replay buffer and episode index from disk."""
    chk = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    print("Checkpoint keys:", list(chk.keys()))

    # model weights
    if 'policy_state' in chk:
        policy_state = chk['policy_state']
    elif 'model_state' in chk:
        policy_state = chk['model_state']
    else:
        raise KeyError("No model weights found in checkpoint.")

    policy_net = DQN(obs_shape, n_actions).to(device)
    res = policy_net.load_state_dict(policy_state, strict=False)
    print(f"  → Missing keys: {res.missing_keys}")
    print(f"  → Unexpected keys: {res.unexpected_keys}")

    # optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    try:
        if 'optimizer_state' in chk:
            optimizer.load_state_dict(chk['optimizer_state'])
        elif 'optim_state' in chk:
            optimizer.load_state_dict(chk['optim_state'])
    except ValueError as e:
        print(f"Warning: optimizer state mismatch ({e}); starting new optimizer.")

    # replay buffer & resume episode
    replay_buffer = chk.get('replay_buffer', ReplayBuffer(BUFFER_CAPACITY))
    start_episode = chk.get('episode', 0) + 1
    print(f"Resuming from episode {start_episode}")

    return policy_net, optimizer, replay_buffer, start_episode


def train():
    # if starting fresh, clear the rewards CSV
    if not RESUME_TRAINING and os.path.exists(REWARDS_CSV):
        os.remove(REWARDS_CSV)

    env = TetrisEnv()
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n

    # resume or fresh
    if RESUME_TRAINING and os.path.exists(CHECKPOINT_PATH):
        policy_net, optimizer, replay_buffer, start_ep = \
            load_checkpoint(obs_shape, n_actions)
        end_ep = start_ep + MAX_EPISODES
    else:
        policy_net = DQN(obs_shape, n_actions).to(device)
        optimizer = optim.Adam(policy_net.parameters(), lr=LR)
        replay_buffer = ReplayBuffer(BUFFER_CAPACITY)
        start_ep = 1
        end_ep = MAX_EPISODES + 1
        print(f"Starting fresh training for {MAX_EPISODES} episodes")

    # target network
    target_net = DQN(obs_shape, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    steps_done = 0

    # main training loop
    for ep in range(start_ep, end_ep):
        env.game.game_over = False
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            eps = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
            steps_done += 1

            if np.random.rand() < eps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_vals = policy_net(torch.tensor(state).unsqueeze(0))
                    action = int(q_vals.argmax())

            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            replay_buffer.push((state, action, reward, next_state, done))
            state = next_state

            # Wait for x milliseconds to better interpret AI actions
            time.delay(0)

            if len(replay_buffer) >= BATCH_SIZE:
                s, a, r, s2, d = replay_buffer.sample(BATCH_SIZE)
                states = torch.tensor(s).to(device)
                actions = torch.tensor(a, dtype=torch.int64).unsqueeze(1).to(device)
                rewards = torch.tensor(r, dtype=torch.float32).to(device)
                next_states = torch.tensor(s2).to(device)
                dones = torch.tensor(d, dtype=torch.bool).to(device)

                q_values = policy_net(states).gather(1, actions).squeeze()
                with torch.no_grad():
                    next_q = target_net(next_states).max(1)[0]
                target = rewards + (~dones) * GAMMA * next_q

                loss = nn.functional.mse_loss(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # end of episode updates
        if ep % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
        if ep % 10 == 0:
            print(f"Episode {ep:>6}, Total Reward: {total_reward:>6}")

        # log & checkpoint
        with open(REWARDS_CSV, 'a') as f:
            f.write(f"{ep},{total_reward}\n")
        if ep % 50 == 0:
            torch.save({
                'episode': ep,
                'model_state': policy_net.state_dict(),
                'optim_state': optimizer.state_dict(),
                'replay_buffer': replay_buffer
            }, CHECKPOINT_PATH)

    env.close()


if __name__ == '__main__':
    train()
