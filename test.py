import torch.nn as nn
import torch
import numpy as np
import gymnasium as gym
import torch.nn.functional as F

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Define the Actor network class
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()

        # Make a simple 3 later linear network
        self.l1 = nn.Linear(24, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 4)
    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        x = torch.tanh(self.l3(x))
        return x
    
# Function to select action from the Actor model given the current state
def select_action(now_state, actor):
    with torch.no_grad():
        now_state = torch.FloatTensor(now_state).to(device)  # Convert state to tensor
        action = actor(now_state).cpu().numpy()  # Get action from actor network
    return action

# Create the environment with human rendering mode
env = gym.make('BipedalWalker-v3', render_mode="human")

# Seed everything for reproducibility
env_seed = 0
torch.manual_seed(0)
torch.cuda.manual_seed(0)

# Initialize the Actor network and move it to the device (GPU or CPU)
actor = Actor().to(device)

# Load the pre-trained model weights for the Actor network
actor.load_state_dict(torch.load("your model name")) # like "BipedalWalker280.pt"

# Switch the Actor network to evaluation mode 
actor.eval()

run = 0  # Initialize episode counter
while run < 3:  # Run for 3 episodes
    # Reset the environment and get the initial state
    now_state = env.reset(seed=env_seed)[0]
    step = 0 
    done = False

    # Interact with the environment 
    while not done:  
        action = select_action(now_state, actor)  # Select action using the actor network
        next_state, reward, done, truncated, info = env.step(action)  # Take the selected action in the environment
        step += 1
        now_state = next_state  # Update the current state to the next state
    print(step)
