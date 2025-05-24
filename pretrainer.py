import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from train_dqn import DQNCNN, device  # import your model and device config
from gym import spaces
import numpy as np

# Load demonstration data
with open("tetris_demonstrations.pkl", "rb") as f:
    demo_data = pickle.load(f)

# Dataset for CNN input
class DemoDataset(Dataset):
    def __init__(self, data):
        self.data = data  # list of (state, action)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state, action = self.data[idx]

        # Ensure CNN input shape: (1, 20, 10)
        state_tensor = torch.tensor(state, dtype=torch.float32)
        return state_tensor, torch.tensor(action, dtype=torch.long)

dataset = DemoDataset(demo_data)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Infer input shape and action space size from data
example_state, example_action = demo_data[0]
obs_shape = spaces.Box(
            low=0, high=1,
            shape=(3, 20, 10),
            dtype=np.int8
        ).shape
n_actions = spaces.Discrete(6).n

# CNN model
policy_net = DQNCNN(obs_shape, n_actions).to(device)

# Wrap for supervised imitation training
class ImitationModel(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.q_network = original_model

    def forward(self, x):
        return self.q_network(x)

model = ImitationModel(policy_net)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Pretrain loop
for epoch in range(100):
    total_loss = 0
    for states, actions in dataloader:
        states = states.to(device)
        actions = actions.to(device)

        logits = model(states)  # shape (B, n_actions)
        loss = criterion(logits, actions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

# Optionally save pretrained model
torch.save(policy_net.state_dict(), "pretrained_dqn.pth")
