# debug_checkpoint.py
# Quick script to see what's in your saved checkpoint

import torch
import os
import numpy as np

CHECKPOINT_PATH = 'dqn_features_checkpoint.pth'

# Need to define ReplayBuffer class so torch.load can deserialize it
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

if os.path.exists(CHECKPOINT_PATH):
    print(f"📂 Found checkpoint: {CHECKPOINT_PATH}")
    print(f"📏 File size: {os.path.getsize(CHECKPOINT_PATH) / 1024:.1f} KB")
    
    try:
        # Load the checkpoint
        checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
        
        print(f"\n📊 Checkpoint contents:")
        for key, value in checkpoint.items():
            if key == 'model_state':
                print(f"  ✅ {key}: {type(value)} (model weights)")
                print(f"      → Contains {len(value)} weight tensors")
            elif key == 'optim_state':
                print(f"  📈 {key}: {type(value)} (optimizer state)")
            elif key == 'replay_buffer':
                print(f"  🎬 {key}: {type(value)} (experience buffer)")
                if hasattr(value, 'buffer'):
                    print(f"      → Buffer size: {len(value.buffer)}")
            elif key == 'episode':
                print(f"  🎯 {key}: {value} (training episodes completed)")
            else:
                print(f"  ❓ {key}: {type(value)}")
        
        print(f"\n✅ Checkpoint looks valid!")
        
        # Check if model weights are reasonable
        if 'model_state' in checkpoint:
            model_weights = checkpoint['model_state']
            first_layer_key = list(model_weights.keys())[0]
            first_weights = model_weights[first_layer_key]
            print(f"\n🔍 First layer weights shape: {first_weights.shape}")
            print(f"🔍 Weight range: {first_weights.min():.3f} to {first_weights.max():.3f}")
            
            if first_weights.shape[1] == 4:
                print("✅ Model expects 4 input features (correct!)")
            else:
                print(f"⚠️  Model expects {first_weights.shape[1]} input features (expected 4)")
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
else:
    print(f"❌ Checkpoint not found: {CHECKPOINT_PATH}")
    print("\nAvailable .pth files:")
    pth_files = [f for f in os.listdir('.') if f.endswith('.pth')]
    if pth_files:
        for f in pth_files:
            print(f"  - {f}")
    else:
        print("  (no .pth files found)")