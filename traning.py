import torch.nn.functional as F
import torch.nn as nn
import torch
import copy
import numpy as np
import gymnasium as gym

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define Actor network for the policy
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(24, 400)  # First hidden layer
        self.l2 = nn.Linear(400, 300)  # Second hidden layer
        self.l3 = nn.Linear(300, 4)     # Output layer

    def forward(self, state):
        # Forward pass through the network
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        return torch.tanh(self.l3(x))  # Tanh activation for output

# Define Critic network for estimating Q-values
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        # Q1 network layers
        self.l1 = nn.Linear(24 + 4, 400)  # State + Action
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        # Q2 network layers (duplicate structure)
        self.l4 = nn.Linear(24 + 4, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)

    def forward(self, state, action): 
        sa = torch.cat([state, action], dim=1)  # Concatenate state and action

        # Forward pass through Q1
        c1 = F.relu(self.l1(sa))
        c1 = F.relu(self.l2(c1))
        c1 = self.l3(c1)

        # Forward pass through Q2
        c2 = F.relu(self.l4(sa))
        c2 = F.relu(self.l5(c2))
        c2 = self.l6(c2)

        return c1, c2  # Return both Q-values

# Replay Buffer for storing experience tuples (state, action, reward, next_state)
class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size):
        self.max_size = max_size  # Maximum size of the buffer
        self.mem_cntr = 0  # Counter for adding new experiences

        # Allocate memory for storing experiences
        self.now_state = torch.zeros((max_size, state_dim), dtype=torch.float).to(device)
        self.action = torch.zeros((max_size, action_dim), dtype=torch.float).to(device)
        self.reward = torch.zeros((max_size, 1), dtype=torch.float).to(device)
        self.next_state = torch.zeros((max_size, state_dim), dtype=torch.float).to(device)
        self.done = torch.zeros((max_size, 1), dtype=torch.float).to(device)

    # Add a new experience to the buffer
    def add(self, now_state, action, reward, next_state, done):
        index = self.mem_cntr % self.max_size  # Find index to store the experience
        self.now_state[index] = torch.from_numpy(now_state).to(device)
        self.action[index] = torch.from_numpy(action).to(device)
        self.next_state[index] = torch.from_numpy(next_state).to(device)
        self.reward[index] = reward
        self.done[index] = float(done)

        self.mem_cntr += 1  # Increment memory counter

    # Sample a batch of experiences from the buffer
    def sample(self, batch_size):
        batch = torch.randint(0, min(self.mem_cntr, self.max_size), (batch_size,)).to(device)  # Randomly sample indices

        return (self.now_state[batch], self.action[batch], 
                self.reward[batch], self.next_state[batch], 
                self.done[batch])

# Training function for both actor and critic networks
def train(actor, actor_optimizer, actor_target, q_critic, q_critic_optimizer, q_critic_target, replay_buffer, step_count):
    batch_size = 128  # Batch size for sampling from replay buffer
    gamma = 0.99
    tau = 0.005  # Soft update factor for target networks

    # Compute the target Q-value
    with torch.no_grad():
        now_state, action, reward, next_state, done = replay_buffer.sample(batch_size)  # Sample from replay buffer
        target_a_next = actor_target(next_state)  # Get action from target actor
        target_Q1, target_Q2 = q_critic_target(next_state, target_a_next)  # Get Q-value from target critic
        target_Q = reward + (1 - done) * gamma * torch.min(target_Q1, target_Q2)  # Calculate target Q-value

    # Get current Q-value estimates from critic
    current_Q1, current_Q2 = q_critic(now_state, action)

    # Compute critic loss using Mean Squared Error (MSE)
    q_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

    # Optimize the critic network
    q_critic_optimizer.zero_grad()
    q_loss.backward()  # Backpropagate the loss
    q_critic_optimizer.step()

    # Update the Actor network every 2 steps
    if step_count % 2 == 0:
        a_loss = -q_critic(now_state, actor(now_state))[0].mean()  # Actor loss (maximize Q-value)
        actor_optimizer.zero_grad()
        a_loss.backward()  # Backpropagate the actor loss
        actor_optimizer.step()

        # Soft update for the target networks (Actor and Critic)
        with torch.no_grad():
            for param, target_param in zip(q_critic.parameters(), q_critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)  # Update target critic

            for param, target_param in zip(actor.parameters(), actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)  # Update target actor

# Function to select action based on the current state and the actor network
def select_action(now_state, actor, deterministic):
    noise_level = 0.1  # Noise level for exploration
    with torch.no_grad():
        now_state = torch.FloatTensor(now_state).to(device)  # Convert state to tensor
        action = actor(now_state).cpu().numpy()  # Get action from actor network
    if deterministic:
        return action
    # Add Gaussian noise for exploration
    noise = np.random.normal(0, noise_level, size=action.shape)  
    action += noise  # Add noise to the action

    # Clip the action to ensure it stays within the valid range
    return np.clip(action, -1, 1)

# Main training loop
def runs():
    # Build Environment
    env = gym.make("BipedalWalker-v3")  # Create environment
    state_dim = 24
    action_dim = 4

    # Seed Everything for reproducibility
    env_seed = 0
    torch.manual_seed(env_seed)
    np.random.seed(env_seed)

    # Initialize Actor network and its optimizer
    actor = Actor().to(device)
    actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=0.001)
    actor_target = copy.deepcopy(actor)  # Create a target actor network

    # Initialize Critic network and its optimizer
    q_critic = Critic().to(device)
    q_critic_optimizer = torch.optim.AdamW(q_critic.parameters(), lr=0.001)
    q_critic_target = copy.deepcopy(q_critic)  # Create a target critic network

    # Initialize Replay Buffer
    replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=int(5e5))

    total_steps = 0  # Total interaction steps
    run = 0  # Number of episodes

    # Training loop
    while run < 1000:
        done = False
        run_reward = 0
        step = 0
        now_state = env.reset(seed=env_seed)[0]  # Reset the environment
        run += 1  # Increment episode counter

        # Interact with the environment and train the networks
        while not done:
            if total_steps < 1000:
                action = env.action_space.sample()  # Random action for initial exploration
            else:
                action = select_action(now_state, actor, deterministic=False)  # Select action from actor

            # Step in the environment
            next_state, reward, done, truncated, info = env.step(action)  
            step += 1  
            total_steps += 1# Increment total steps

            if step >= 2000:
                done = True
                reward -= 100
            run_reward += reward  # Accumulate reward

            # Add experience to replay buffer
            replay_buffer.add(now_state, action, reward, next_state, done)  
            now_state = next_state  # Update current state
            

            if total_steps > 1000:
                train(actor, actor_optimizer, actor_target, q_critic, q_critic_optimizer, q_critic_target, replay_buffer, total_steps)
        if run_reward > 200:
            env.close()
            test_for_save(actor)
        print(f"Steps: {step}, Episode: {run}, Reward: {run_reward}")  # Print episode statistics

    env.close()  # Close the environment
    return actor, run_reward

def test_for_save(actor):
    average_save_rewards = 200  # Threshold for saving the model
    env_test = gym.make("BipedalWalker-v3")  # Create test environment
    run_reward = 0
    done = False
    now_state = env_test.reset(seed=0)[0]  # Reset environment and get initial state
    while not done:
        action = select_action(now_state, actor, deterministic=True)  # Select action
        next_state, reward, done, truncated, info = env_test.step(action)  # Take action in environment
        run_reward += reward  # Accumulate reward
        now_state = next_state  # Update state
    print(f"test reward = {run_reward}")  # Print the test reward
    if run_reward >= average_save_rewards:
        average_save_rewards = run_reward  # Update the threshold
        torch.save(actor.state_dict(), f"BipedalWalker{int(run_reward)}.pt")  # Save the model if reward is high enough
    env_test.close()

while True :
    actor, run_reward = runs()
    if run_reward > 0:
        break
torch.save(actor.state_dict(), f"BipedalWalker-v3.pt")  # Save the model if reward is high enough