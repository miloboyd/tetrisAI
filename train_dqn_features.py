# train_dqn_features_fixed.py
# FIXED: Training with properly aligned reward system

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tetris_env_features import TetrisEnvFeaturesFixed
import matplotlib.pyplot as plt
import time

# Hyperparameters - optimized for long training with fixed rewards
LR               = 3e-4    # Slightly lower for stability
GAMMA            = 0.99    
EPS_START        = 1.0
EPS_END          = 0.02    # Keep some exploration
EPS_DECAY        = 8000    # Decay over 8k episodes
BATCH_SIZE       = 64      # Larger batches for stability
BUFFER_CAPACITY  = 50000   # Larger buffer for more diverse experiences
TARGET_UPDATE    = 20      # Less frequent updates for stability
CHECKPOINT_PATH  = 'dqn_fixed_checkpoint.pth'
REWARDS_CSV      = 'training_rewards_fixed.csv'

MAX_EPISODES     = 10000   # Long training run
MAX_STEPS_PER_EPISODE = 2000  # Prevent infinite games
RESUME_TRAINING  = True    # Enable resuming for Colab
RENDERGAME       = False   # CRITICAL: No rendering for speed

# Progress tracking
PRINT_INTERVAL   = 50      # Print every 50 episodes
SAVE_INTERVAL    = 200     # Save every 200 episodes
PLOT_INTERVAL    = 500     # Update plots every 500 episodes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 FIXED TRAINING - Using device: {device}")

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

class SimpleDQN(nn.Module):
    def __init__(self, feature_size=4, n_actions=6):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(feature_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions)
        )

    def forward(self, x):
        x = x.to(device).float()
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        return self.model(x)

def load_checkpoint():
    """Load checkpoint for resuming training"""
    try:
        chk = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        print("📁 Checkpoint keys:", list(chk.keys()))

        policy_net = SimpleDQN().to(device)
        policy_net.load_state_dict(chk['model_state'])

        optimizer = optim.Adam(policy_net.parameters(), lr=LR)
        try:
            optimizer.load_state_dict(chk['optimizer_state'])
        except:
            print("⚠️  Warning: Could not load optimizer state, starting fresh optimizer")

        # Create new replay buffer instead of loading old one (saves memory)
        replay_buffer = ReplayBuffer(BUFFER_CAPACITY)
        
        start_episode = chk.get('episode', 0) + 1
        best_score = chk.get('best_score', 0)
        
        print(f"✅ Resuming from episode {start_episode}, best score: {best_score}")
        return policy_net, optimizer, replay_buffer, start_episode, best_score
        
    except FileNotFoundError:
        print("📝 No checkpoint found, starting fresh")
        return None, None, None, 1, 0
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return None, None, None, 1, 0

def plot_training_progress(episode_rewards, episode_scores, episode_score_improvements, episode):
    """Plot training progress with FIXED metrics"""
    if len(episode_rewards) < 10:
        return
        
    plt.figure(figsize=(20, 10))
    
    # Rewards plot
    plt.subplot(2, 4, 1)
    plt.plot(episode_rewards[-1000:])
    plt.title(f'Episode Rewards (Last 1000)\nEpisode {episode}')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    
    # Scores plot
    plt.subplot(2, 4, 2)
    plt.plot(episode_scores[-1000:])
    plt.title('Episode Scores (Last 1000)')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.grid(True)
    
    # NEW: Score improvements plot
    plt.subplot(2, 4, 3)
    if len(episode_score_improvements) > 0:
        plt.plot(episode_score_improvements[-1000:])
        plt.title('Score Improvements per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Score Improvement')
        plt.grid(True)
    
    # Moving averages - rewards
    plt.subplot(2, 4, 4)
    if len(episode_rewards) >= 100:
        reward_ma = np.convolve(episode_rewards, np.ones(100)/100, mode='valid')
        plt.plot(reward_ma[-1000:], label='Reward (100-ep avg)', alpha=0.7)
        plt.title('Reward Moving Average')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.legend()
        plt.grid(True)
    
    # Moving averages - scores
    plt.subplot(2, 4, 5)
    if len(episode_scores) >= 100:
        score_ma = np.convolve(episode_scores, np.ones(100)/100, mode='valid')
        plt.plot(score_ma[-1000:], label='Score (100-ep avg)', alpha=0.7, color='orange')
        plt.title('Score Moving Average')
        plt.xlabel('Episode')
        plt.ylabel('Average Score')
        plt.legend()
        plt.grid(True)
    
    # Reward vs Score correlation
    plt.subplot(2, 4, 6)
    if len(episode_rewards) > 50:
        recent_rewards = episode_rewards[-1000:]
        recent_scores = episode_scores[-1000:]
        plt.scatter(recent_scores, recent_rewards, alpha=0.6, s=1)
        plt.title('Reward vs Score Correlation')
        plt.xlabel('Score')
        plt.ylabel('Reward')
        plt.grid(True)
    
    # Best scores over time
    plt.subplot(2, 4, 7)
    if len(episode_scores) > 0:
        best_scores_so_far = []
        current_best = 0
        for score in episode_scores:
            if score > current_best:
                current_best = score
            best_scores_so_far.append(current_best)
        plt.plot(best_scores_so_far[-1000:])
        plt.title('Best Score Over Time')
        plt.xlabel('Episode')
        plt.ylabel('Best Score So Far')
        plt.grid(True)
    
    # Episode statistics
    plt.subplot(2, 4, 8)
    recent_data = {
        'Avg Reward': np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else 0,
        'Avg Score': np.mean(episode_scores[-100:]) if len(episode_scores) >= 100 else 0,
        'Max Score': max(episode_scores[-100:]) if len(episode_scores) >= 100 else 0,
        'Current Best': max(episode_scores) if episode_scores else 0
    }
    
    bars = plt.bar(recent_data.keys(), recent_data.values())
    plt.title('Recent Performance (Last 100 episodes)')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, recent_data.values()):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(recent_data.values())*0.01,
                f'{value:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def train():
    print("🚀 Starting FIXED Tetris DQN training")
    print("🎯 Goal: Reward system now aligned with Tetris score!")
    print(f"📊 Target: {MAX_EPISODES} episodes")
    print(f"🏗️  Network: 4 → 32 → 32 → 6")
    print(f"💾 Device: {device}")
    print("🔧 Key fix: Reward = score improvement + line bonuses, NOT just survival")
    
    # Initialize environment (no rendering)
    env = TetrisEnvFeaturesFixed()  # Use the FIXED environment
    env.pyRender(RENDERGAME)
    
    # Try to resume training
    if RESUME_TRAINING:
        policy_net, optimizer, replay_buffer, start_ep, best_score = load_checkpoint()
    else:
        policy_net, optimizer, replay_buffer, start_ep, best_score = None, None, None, 1, 0
    
    # Create fresh training setup if needed
    if policy_net is None:
        policy_net = SimpleDQN().to(device)
        optimizer = optim.Adam(policy_net.parameters(), lr=LR)
        replay_buffer = ReplayBuffer(BUFFER_CAPACITY)
        start_ep = 1
        best_score = 0
        print("🆕 Starting fresh training with FIXED reward system")

    # Target network
    target_net = SimpleDQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # Progress tracking
    episode_rewards = []
    episode_scores = []
    episode_score_improvements = []  # NEW: Track score improvements
    episode_steps_list = []
    start_time = time.time()
    steps_done = 0

    # Clear CSV if starting fresh
    if start_ep == 1 and os.path.exists(REWARDS_CSV):
        os.remove(REWARDS_CSV)

    # Training loop
    for ep in range(start_ep, MAX_EPISODES + 1):
        # Reset environment properly
        state = env.reset()
        env.game.game_over = False  # Ensure game_over is reset
        
        total_reward = 0
        total_score_improvement = 0  # NEW: Track per episode
        episode_steps = 0
        done = False

        while not done and episode_steps < MAX_STEPS_PER_EPISODE:
            # Epsilon-greedy action selection
            epsilon = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
            steps_done += 1

            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_vals = policy_net(torch.tensor(state, dtype=torch.float32))
                    action = int(q_vals.argmax())

            # Take action
            try:
                next_state, reward, done, info = env.step(action)
                total_reward += reward
                
                # NEW: Track score improvement
                score_improvement = info.get('score_improvement', 0)
                total_score_improvement += score_improvement
                
                episode_steps += 1
                
                replay_buffer.push((state, action, reward, next_state, done))
                state = next_state
            except Exception as e:
                print(f"❌ Error in episode {ep}, step {episode_steps}: {e}")
                done = True
                break

            # Training step
            if len(replay_buffer) >= BATCH_SIZE and episode_steps % 4 == 0:
                try:
                    s, a, r, s2, d = replay_buffer.sample(BATCH_SIZE)
                    
                    states = torch.tensor(s, dtype=torch.float32).to(device)
                    actions = torch.tensor(a, dtype=torch.int64).unsqueeze(1).to(device)
                    rewards = torch.tensor(r, dtype=torch.float32).to(device)
                    next_states = torch.tensor(s2, dtype=torch.float32).to(device)
                    dones = torch.tensor(d, dtype=torch.bool).to(device)

                    # Current Q values
                    q_values = policy_net(states).gather(1, actions).squeeze()
                    
                    # Target Q values
                    with torch.no_grad():
                        next_actions = policy_net(next_states).argmax(1)
                        next_q = target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
                        target = rewards + (~dones) * GAMMA * next_q

                    # Compute loss and update
                    loss = nn.functional.mse_loss(q_values, target)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                    optimizer.step()
                    
                except Exception as e:
                    print(f"⚠️  Training error in episode {ep}: {e}")

        # Update target network
        if ep % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
        # Track statistics
        current_score = info.get('score', 0) if 'info' in locals() else 0
        if current_score > best_score:
            best_score = current_score
            
        episode_rewards.append(total_reward)
        episode_scores.append(current_score)
        episode_score_improvements.append(total_score_improvement)  # NEW
        episode_steps_list.append(episode_steps)

        # Progress reporting
        if ep % PRINT_INTERVAL == 0:
            elapsed = time.time() - start_time
            eps_per_min = ep / (elapsed / 60) if elapsed > 0 else 0
            avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            avg_score = np.mean(episode_scores[-100:]) if len(episode_scores) >= 100 else np.mean(episode_scores)
            avg_score_improvement = np.mean(episode_score_improvements[-100:]) if len(episode_score_improvements) >= 100 else np.mean(episode_score_improvements)
            avg_steps = np.mean(episode_steps_list[-100:]) if len(episode_steps_list) >= 100 else np.mean(episode_steps_list)
            
            print(f"Episode {ep:5d} | Reward: {total_reward:6.1f} | Score: {current_score:3d} | "
                  f"Best: {best_score:3d} | Eps: {epsilon:.3f}")
            print(f"  Avg (100): Reward {avg_reward:6.1f} | Score {avg_score:5.1f} | "
                  f"Score⬆ {avg_score_improvement:5.1f} | Steps {avg_steps:5.1f} | {eps_per_min:.1f} ep/min")

        # Save progress
        with open(REWARDS_CSV, 'a') as f:
            f.write(f"{ep},{total_reward},{current_score},{episode_steps},{total_score_improvement}\n")
            
        # Save checkpoint (without replay buffer to save space)
        if ep % SAVE_INTERVAL == 0:
            torch.save({
                'episode': ep,
                'model_state': policy_net.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_score': best_score
            }, CHECKPOINT_PATH)
            print(f"    💾 Checkpoint saved at episode {ep} (Best score: {best_score})")
        
        # Plot progress
        if ep % PLOT_INTERVAL == 0:
            plot_training_progress(episode_rewards, episode_scores, episode_score_improvements, ep)

    # Final save
    torch.save({
        'episode': MAX_EPISODES,
        'model_state': policy_net.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'best_score': best_score
    }, CHECKPOINT_PATH)
    
    print(f"\n🏁 Training complete!")
    print(f"📊 Best score achieved: {best_score}")
    print(f"⏱️  Total time: {(time.time() - start_time) / 3600:.1f} hours")
    print(f"🎯 Reward system was FIXED - should see much better score progression!")
    
    # Final plot
    plot_training_progress(episode_rewards, episode_scores, episode_score_improvements, MAX_EPISODES)

    env.close()

if __name__ == '__main__':
    print("=" * 60)
    print("🔧 FIXED TETRIS DQN TRAINING")
    print("=" * 60)
    print("✅ Reward now prioritizes SCORE IMPROVEMENT over survival")
    print("✅ AI will learn to maximize Tetris score, not just survive")
    print("✅ Enhanced monitoring of score improvements")
    print("=" * 60)
    train()