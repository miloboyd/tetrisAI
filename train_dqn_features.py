# train_dqn_features.py
# Modified version of your train_dqn.py for feature-based approach

import os
import numpy as np
import pygame
from pygame import time
import torch
import torch.nn as nn
import torch.optim as optim
from tetris_env_features import TetrisEnvFeatures  # NEW: Use feature-based env

# Keep your existing hyperparameters but adjust some for faster training
LR               = 1e-3
GAMMA            = 0.995
EPS_START        = 1.0
EPS_END          = 0.1
EPS_DECAY        = 10000  # Faster decay since features learn quicker
BATCH_SIZE       = 64
BUFFER_CAPACITY  = 50000   # Smaller buffer is fine
TARGET_UPDATE    = 5       # More frequent updates for stability
CHECKPOINT_PATH  = 'dqn_features_checkpoint.pth'
REWARDS_CSV      = 'training_rewards_features.csv'

MAX_EPISODES     = 1000    # Should see good results much sooner
RESUME_TRAINING  = False   
RENDERGAME       = False   

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Keep your existing ReplayBuffer class
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

# NEW: Simple DQN for feature input instead of CNN
class FeatureDQN(nn.Module):
    def __init__(self, feature_size, n_actions):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        x = x.to(device).float()
        if len(x.shape) == 1:  # Single sample
            x = x.unsqueeze(0)
        return self.model(x)

# Keep most of your existing checkpoint loading logic but adapt for features
def load_checkpoint(feature_size, n_actions):
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

    policy_net = FeatureDQN(feature_size, n_actions).to(device)
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
    # Clear CSV if starting fresh
    if not RESUME_TRAINING and os.path.exists(REWARDS_CSV):
        os.remove(REWARDS_CSV)

    # NEW: Use feature-based environment
    env = TetrisEnvFeatures()
    env.pyRender(RENDERGAME)
    
    # NEW: Get feature size from environment (should be 4 now)
    feature_size = env.observation_space.shape[0]  # Should be 4
    n_actions = env.action_space.n  # Should be 6
    
    print(f"Feature size: {feature_size} (should be 4)")
    print(f"Action space: {n_actions}")

    # Resume or start fresh
    if RESUME_TRAINING and os.path.exists(CHECKPOINT_PATH):
        policy_net, optimizer, replay_buffer, start_ep = \
            load_checkpoint(feature_size, n_actions)
        end_ep = start_ep + MAX_EPISODES
    else:
        policy_net = FeatureDQN(feature_size, n_actions).to(device)
        optimizer = optim.Adam(policy_net.parameters(), lr=LR)
        replay_buffer = ReplayBuffer(BUFFER_CAPACITY)
        start_ep = 1
        end_ep = MAX_EPISODES + 1
        print(f"Starting fresh feature-based training for {MAX_EPISODES} episodes")

    # target network
    target_net = FeatureDQN(feature_size, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    steps_done = 0

    # Main training loop (keep your existing structure)
    for ep in range(start_ep, end_ep):
        env.game.game_over = False
        state = env.reset()  # Now returns feature vector
        total_reward = 0
        done = False

        while not done:
            if env.render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done = True
                        break
                
            epsilon = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
            steps_done += 1

            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    # State is now a feature vector, not an image
                    q_vals = policy_net(torch.tensor(state, dtype=torch.float32))
                    action = int(q_vals.argmax())

            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            replay_buffer.push((state, action, reward, next_state, done))
            state = next_state

            # Training step
            if len(replay_buffer) >= BATCH_SIZE:
                s, a, r, s2, d = replay_buffer.sample(BATCH_SIZE)
                states = torch.tensor(s, dtype=torch.float32).to(device)
                actions = torch.tensor(a, dtype=torch.int64).unsqueeze(1).to(device)
                rewards = torch.tensor(r, dtype=torch.float32).to(device)
                next_states = torch.tensor(s2, dtype=torch.float32).to(device)
                dones = torch.tensor(d, dtype=torch.bool).to(device)

                q_values = policy_net(states).gather(1, actions).squeeze()
                with torch.no_grad():
                    # Double DQN
                    next_actions = policy_net(next_states).argmax(1)
                    next_q = target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
                target = rewards + (~dones) * GAMMA * next_q

                loss = nn.functional.mse_loss(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                
                # Add gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()

        # Update target network more frequently
        if ep % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
        if ep % 10 == 0:
            print(f"Episode {ep:<5}, Total Reward: {total_reward:.2f}, Score: {env.game.score}, Epsilon: {epsilon:.3f}")

        # Save progress
        with open(REWARDS_CSV, 'a') as f:
            f.write(f"{ep},{total_reward},{env.game.score}\n")
            
        if ep % 50 == 0:  # Save more frequently to track progress
            torch.save({
                'episode': ep,
                'model_state': policy_net.state_dict(),
                'optim_state': optimizer.state_dict(),
                'replay_buffer': replay_buffer
            }, CHECKPOINT_PATH)

    env.close()

if __name__ == '__main__':
    print("Starting feature-based Tetris DQN training...")
    train()