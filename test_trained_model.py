# test_trained_model.py
# Fixed version that matches your training setup exactly

import os
import numpy as np
import pygame
from pygame import time
import torch
import torch.nn as nn
from tetris_env_features import TetrisEnvFeatures

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = 'dqn_features_checkpoint.pth'

# Need ReplayBuffer class definition for loading checkpoint
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

# EXACT same model as your training script
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

def test_model():
    print("🎮 Testing Your Trained Tetris AI")
    
    # Setup environment exactly like training
    env = TetrisEnvFeatures()
    env.pyRender(True)  # Enable visual rendering
    
    feature_size = 4  # Your model uses 4 features
    n_actions = 6     # Your model uses 6 actions
    
    print(f"Loading model with {feature_size} features and {n_actions} actions")
    
    # Create model with exact same architecture
    model = FeatureDQN(feature_size, n_actions).to(device)
    
    # Load the checkpoint
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ Checkpoint not found: {CHECKPOINT_PATH}")
        print("Make sure you've completed training first!")
        return
    
    try:
        print(f"📂 Loading checkpoint from {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        
        # Debug: Print what's in the checkpoint
        print(f"📊 Checkpoint keys: {list(checkpoint.keys())}")
        
        # Load the model state (use correct key from your training)
        if 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
            episode = checkpoint.get('episode', 'unknown')
            print(f"✅ Model loaded successfully from episode {episode}")
        else:
            print(f"❌ No 'model_state' found in checkpoint. Available keys: {list(checkpoint.keys())}")
            return
            
        model.eval()  # Set to evaluation mode
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return
    
    print("\n🚀 Starting AI gameplay...")
    print("📝 Watch the AI play! Features will be printed every 50 steps.")
    print("🔄 Close the window to exit.")
    
    game_number = 0
    
    try:
        while True:
            game_number += 1
            print(f"\n🎯 === Game {game_number} ===")
            
            # Reset for new game
            state = env.reset()
            total_reward = 0
            steps = 0
            done = False
            
            print(f"Starting features: {state}")
            
            while not done:
                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("👋 Exiting...")
                        env.close()
                        return
                
                # AI decision making (NO randomness - pure learned policy)
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32)
                    q_values = model(state_tensor)
                    action = int(q_values.argmax().item())  # Best action
                    
                    # Debug: Print AI's thinking every 50 steps
                    if steps % 50 == 0:
                        action_names = ["Left", "Right", "Soft Drop", "Hard Drop", "Rotate CW", "Rotate CCW"]
                        q_vals_cpu = q_values.cpu().numpy().flatten()
                        best_action_name = action_names[action]
                        print(f"  Step {steps:3d}: Features={state} → Action='{best_action_name}'")
                        print(f"           Q-values: {q_vals_cpu}")
                
                # Take the action
                next_state, reward, done, info = env.step(action)
                total_reward += reward
                steps += 1
                state = next_state
                
                # Add delay so we can watch (adjust for speed)
                time.wait(100)  # 100ms delay - make smaller for faster gameplay
            
            # Game finished - print results
            final_score = info['score']
            print(f"\n🏁 Game {game_number} Results:")
            print(f"   Final Score: {final_score}")
            print(f"   Total Steps: {steps}")
            print(f"   Total Reward: {total_reward:.1f}")
            print(f"   Final Features: {state}")
            
            # Brief pause before next game
            print("   Next game in 3 seconds...")
            time.wait(3000)
    
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during gameplay: {e}")
    finally:
        env.close()

if __name__ == "__main__":
    test_model()