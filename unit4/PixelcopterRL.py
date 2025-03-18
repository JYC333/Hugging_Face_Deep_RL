import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

# Gym
# import gymnasium as gym
import gym
import gym_pygame

# Hugging Face Hub
from utils_gym import reinforce, evaluate_agent, push_to_hub


class Policy(nn.Module):
    def __init__(self, s_size, a_size, h_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, h_size * 2)
        self.fc3 = nn.Linear(h_size * 2, a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env_id = "Pixelcopter-PLE-v0"
# Create the env
env = gym.make(env_id)

# Create the evaluation env
eval_env = gym.make(env_id)

# Get the state space and action space
s_size = env.observation_space.shape[0]
a_size = env.action_space.n

pixelcopter_hyperparameters = {
    "h_size": 64,
    "n_training_episodes": 100000,
    "n_evaluation_episodes": 10,
    "max_t": 10000,
    "gamma": 0.989,
    "lr": 1e-4,
    "env_id": env_id,
    "state_space": s_size,
    "action_space": a_size,
}

# Create policy and place it to the device
# torch.manual_seed(50)
pixelcopter_policy = Policy(
    pixelcopter_hyperparameters["state_space"],
    pixelcopter_hyperparameters["action_space"],
    pixelcopter_hyperparameters["h_size"],
).to(device)
pixelcopter_optimizer = optim.Adam(
    pixelcopter_policy.parameters(), lr=pixelcopter_hyperparameters["lr"]
)

scores = reinforce(
    env,
    pixelcopter_policy,
    pixelcopter_optimizer,
    pixelcopter_hyperparameters["n_training_episodes"],
    pixelcopter_hyperparameters["max_t"],
    pixelcopter_hyperparameters["gamma"],
    1000,
)
print(
    evaluate_agent(
        eval_env,
        pixelcopter_hyperparameters["max_t"],
        pixelcopter_hyperparameters["n_evaluation_episodes"],
        pixelcopter_policy,
    )
)

repo_id = (
    "JYC333/Reinforce-" + env_id
)  # TODO Define your repo id {username/Reinforce-{model-id}}
push_to_hub(
    env,
    env_id,
    repo_id,
    pixelcopter_policy,  # The model we want to save
    pixelcopter_hyperparameters,  # Hyperparameters
    eval_env,  # Evaluation environment
    video_fps=30,
)
