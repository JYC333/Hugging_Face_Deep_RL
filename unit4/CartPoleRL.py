import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

# Gym
import gymnasium as gym

# Hugging Face Hub
from utils import reinforce, evaluate_agent, push_to_hub


class Policy(nn.Module):
    def __init__(self, s_size, a_size, h_size):
        super(Policy, self).__init__()
        # Create two fully connected layers
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)

    def forward(self, x):
        # Define the forward pass
        # state goes to fc1 then we apply ReLU activation function
        x = F.relu(self.fc1(x))
        # fc1 outputs goes to fc2
        x = self.fc2(x)
        # We output the softmax
        return F.softmax(x, dim=1)

    def act(self, state):
        """
        Given a state, take action
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env_id = "CartPole-v1"
# Create the env
env = gym.make(env_id, render_mode="rgb_array")

# Create the evaluation env
eval_env = gym.make(env_id, render_mode="rgb_array")

# Get the state space and action space
s_size = env.observation_space.shape[0]
a_size = int(env.action_space.n)

cartpole_hyperparameters = {
    "h_size": 16,
    "n_training_episodes": 1000,
    "n_evaluation_episodes": 10,
    "max_t": 1000,
    "gamma": 1.0,
    "lr": 1e-2,
    "env_id": env_id,
    "state_space": s_size,
    "action_space": a_size,
}

# Create policy and place it to the device
cartpole_policy = Policy(
    cartpole_hyperparameters["state_space"],
    cartpole_hyperparameters["action_space"],
    cartpole_hyperparameters["h_size"],
).to(device)
cartpole_optimizer = optim.Adam(
    cartpole_policy.parameters(), lr=cartpole_hyperparameters["lr"]
)

scores = reinforce(
    env,
    cartpole_policy,
    cartpole_optimizer,
    cartpole_hyperparameters["n_training_episodes"],
    cartpole_hyperparameters["max_t"],
    cartpole_hyperparameters["gamma"],
    100,
)

print(
    evaluate_agent(
        eval_env,
        cartpole_hyperparameters["max_t"],
        cartpole_hyperparameters["n_evaluation_episodes"],
        cartpole_policy,
    )
)

repo_id = (
    "JYC333/Reinforce-" + env_id
)  # TODO Define your repo id {username/Reinforce-{model-id}}
push_to_hub(
    env,
    env_id,
    repo_id,
    cartpole_policy,  # The model we want to save
    cartpole_hyperparameters,  # Hyperparameters
    eval_env,  # Evaluation environment
    video_fps=30,
)
