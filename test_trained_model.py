# test_simple_model.py
# Test the simple 4→32→32→6 network

import os
import numpy as np
import pygame
from pygame import time
import torch
import torch.nn as nn
from tetris_env_features import TetrisEnvFeatures

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = 'dqn_simple_checkpoint.pth'

# EXACT same simple network as training
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

def test_simple_model():
    print("🎮 Testing Simple Tetris AI (4→32→32→6)")
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ Checkpoint not found: {CHECKPOINT_PATH}")
        print("Run train_dqn_simple.py first!")
        return
    
    # Load model
    model = SimpleDQN().to(device)
    
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        
        episode = checkpoint.get('episode', 'unknown')
        best_score = checkpoint.get('best_score', 'unknown')
        print(f"✅ Loaded simple model from episode {episode}")
        print(f"📊 Best training score: {best_score}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Setup environment
    env = TetrisEnvFeatures()
    env.pyRender(True)
    
    print("\n🚀 Starting simple AI gameplay...")
    print("📝 Simpler network should make more varied decisions")
    print("🔄 Close window to exit")
    
    game_count = 0
    scores = []
    
    try:
        while True:
            game_count += 1
            print(f"\n🎯 === Game {game_count} ===")
            
            state = env.reset()
            total_reward = 0
            steps = 0
            done = False
            
            while not done:
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("👋 Exiting...")
                        env.close()
                        
                        if scores:
                            print(f"\n📊 Final Statistics:")
                            print(f"   Games played: {len(scores)}")
                            print(f"   Average score: {np.mean(scores):.1f}")
                            print(f"   Best score: {max(scores)}")
                            print(f"   Scores: {scores}")
                        return
                
                # AI decision (pure exploitation)
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32)
                    q_values = model(state_tensor)
                    action = int(q_values.argmax().item())
                    
                    # Show AI thinking every 100 steps
                    if steps % 100 == 0:
                        action_names = ["Left", "Right", "Soft Drop", "Hard Drop", "Rotate CW", "Rotate CCW"]
                        q_vals = q_values.cpu().numpy().flatten()
                        best_action = action_names[action]
                        
                        print(f"  Step {steps:3d}: Features=[{state[0]:.0f}, {state[1]:.0f}, {state[2]:.0f}, {state[3]:.0f}] → '{best_action}'")
                        
                        # Show which actions look good/bad
                        action_quality = []
                        for i, (name, q_val) in enumerate(zip(action_names, q_vals)):
                            marker = " ←" if i == action else ""
                            action_quality.append(f"{name}:{q_val:.1f}{marker}")
                        print(f"           Q-values: {', '.join(action_quality)}")
                
                # Take action
                next_state, reward, done, info = env.step(action)
                total_reward += reward
                steps += 1
                state = next_state
                
                # Adjust game speed (lower = faster)
                time.wait(60)
            
            # Game finished
            final_score = info['score']
            scores.append(final_score)
            
            print(f"\n🏁 Game {game_count} Complete:")
            print(f"   Score: {final_score}")
            print(f"   Steps: {steps}")
            print(f"   Reward: {total_reward:.1f}")
            print(f"   Running avg: {np.mean(scores[-10:]):.1f} (last 10 games)")
            
            # Brief pause
            time.wait(1500)
    
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        env.close()

if __name__ == "__main__":
    test_simple_model()