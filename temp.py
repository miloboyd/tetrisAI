import gym
import simple_driving
import torch
import torch.nn as nn
import numpy as np

# Set up the environment with rendering ON to visualize
env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=True, isDiscrete=True)
env = env.unwrapped

state_dim = 8 #2 + 2*env.getObsSize() #env.observation_space.shape[0]
action_dim = env.action_space.n

# Same network architecture as during training
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# Load the trained model
model = DQN(state_dim, action_dim)
#model.load_state_dict(torch.load("30000_epoch_training.pth"))
model.load_state_dict(torch.load("dqn_simple_driving_mid-training.pth"))
model.eval()

def select_action(state):
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = model(state_tensor)
        return q_values.argmax().item()

# Run a few test episodes
num_test_episodes = 5
max_steps = 200

for episode in range(num_test_episodes):
    state, info = env.reset()
    total_reward = 0

    for step in range(max_steps):
        action = select_action(state)
        state, reward, done, _, _ = env.step(action)
        total_reward = reward
        if done:
            break

    print(f"Episode {episode + 1} — Total Reward: {total_reward:.2f}")

env.close()