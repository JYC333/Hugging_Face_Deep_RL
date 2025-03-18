import gymnasium as gym
from utils import push_to_hub, load_from_hub, evaluate_agent
from QLearning import QLearning
from pathlib import Path

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler


def FrozenLake_no_slippery():
    env = gym.make(
        "FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="rgb_array"
    )

    model = QLearning(env)
    # model.train()
    model.train_with_video(Path("./unit2/video"), 30)
    print(model.evaluate_agent())

    model_dict = {
        "env_id": model.env_id,
        "max_steps": model.max_steps,
        "n_training_episodes": model.n_training_episodes,
        "n_eval_episodes": model.n_eval_episodes,
        "eval_seed": model.eval_seed,
        "learning_rate": model.learning_rate,
        "gamma": model.gamma,
        "max_epsilon": model.max_epsilon,
        "min_epsilon": model.min_epsilon,
        "decay_rate": model.decay_rate,
        "qtable": model.Qtable,
    }

    # username = "JYC333"  # FILL THIS
    # repo_name = "q-FrozenLake-v1-4x4-noSlippery"
    # push_to_hub(
    #     env_id=model.env_id,
    #     repo_id=f"{username}/{repo_name}",
    #     model=model_dict,
    #     env=env,
    # )


def Taxi():
    env = gym.make("Taxi-v3", render_mode="rgb_array")

    eval_seed = [
        16,
        54,
        165,
        177,
        191,
        191,
        120,
        80,
        149,
        178,
        48,
        38,
        6,
        125,
        174,
        73,
        50,
        172,
        100,
        148,
        146,
        6,
        25,
        40,
        68,
        148,
        49,
        167,
        9,
        97,
        164,
        176,
        61,
        7,
        54,
        55,
        161,
        131,
        184,
        51,
        170,
        12,
        120,
        113,
        95,
        126,
        51,
        98,
        36,
        135,
        54,
        82,
        45,
        95,
        89,
        59,
        95,
        124,
        9,
        113,
        58,
        85,
        51,
        134,
        121,
        169,
        105,
        21,
        30,
        11,
        50,
        65,
        12,
        43,
        82,
        145,
        152,
        97,
        106,
        55,
        31,
        85,
        38,
        112,
        102,
        168,
        123,
        97,
        21,
        83,
        158,
        26,
        80,
        63,
        5,
        81,
        32,
        11,
        28,
        148,
    ]

    model = QLearning(env, n_training_episodes=100000, eval_seed=eval_seed)

    def objective(trial):
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
        gamma = trial.suggest_float("gamma", 0.9, 0.99999, log=True)
        max_epsilon = trial.suggest_float("max_epsilon", 0.5, 1, log=True)
        min_epsilon = trial.suggest_float("min_epsilon", 1e-5, 0.5, log=True)
        decay_rate = trial.suggest_float("decay_rate", 1e-5, 1, log=True)

        model.learning_rate = learning_rate
        model.gamma = gamma
        model.max_epsilon = max_epsilon
        model.min_epsilon = min_epsilon
        model.decay_rate = decay_rate

        if not model.train(trial=trial):
            raise optuna.exceptions.TrialPruned()

        mean_reward, std_reward = model.evaluate_agent()

        return mean_reward - std_reward

    sampler = TPESampler(n_startup_trials=5)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")
    study.optimize(objective, n_trials=100)

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print(f"Value: {trial.value}")

    print("Params: ")
    for key, value in trial.params.items():
        print(f"{key}: {value}")

    model_dict = {
        "env_id": model.env_id,
        "max_steps": model.max_steps,
        "n_training_episodes": model.n_training_episodes,
        "n_eval_episodes": model.n_eval_episodes,
        "eval_seed": model.eval_seed,
        "learning_rate": trial.params["learning_rate"],
        "gamma": trial.params["gamma"],
        "max_epsilon": trial.params["max_epsilon"],
        "min_epsilon": trial.params["min_epsilon"],
        "decay_rate": trial.params["decay_rate"],
        "qtable": model.Qtable,
    }

    username = "JYC333"  # FILL THIS
    repo_name = "q-Taxi-v3"
    push_to_hub(
        env_id=model.env_id,
        repo_id=f"{username}/{repo_name}",
        model=model_dict,
        env=env,
    )


FrozenLake_no_slippery()

# Taxi()

# model_download = load_from_hub(
#     repo_id="DBusAI/q-Taxi-v3-v5", filename="q-learning.pkl"
# )  # Try to use another model

# eval_seed = [
#     16,
#     54,
#     165,
#     177,
#     191,
#     191,
#     120,
#     80,
#     149,
#     178,
#     48,
#     38,
#     6,
#     125,
#     174,
#     73,
#     50,
#     172,
#     100,
#     148,
#     146,
#     6,
#     25,
#     40,
#     68,
#     148,
#     49,
#     167,
#     9,
#     97,
#     164,
#     176,
#     61,
#     7,
#     54,
#     55,
#     161,
#     131,
#     184,
#     51,
#     170,
#     12,
#     120,
#     113,
#     95,
#     126,
#     51,
#     98,
#     36,
#     135,
#     54,
#     82,
#     45,
#     95,
#     89,
#     59,
#     95,
#     124,
#     9,
#     113,
#     58,
#     85,
#     51,
#     134,
#     121,
#     169,
#     105,
#     21,
#     30,
#     11,
#     50,
#     65,
#     12,
#     43,
#     82,
#     145,
#     152,
#     97,
#     106,
#     55,
#     31,
#     85,
#     38,
#     112,
#     102,
#     168,
#     123,
#     97,
#     21,
#     83,
#     158,
#     26,
#     80,
#     63,
#     5,
#     81,
#     32,
#     11,
#     28,
#     148,
# ]


# print(model_download)
# print(model_download["eval_seed"] == eval_seed)
# env = gym.make(model_download["env_id"], render_mode="rgb_array")

# model = QLearning(
#     env,
#     n_training_episodes=1000000,
#     learning_rate=model_download["learning_rate"],
#     n_eval_episodes=model_download["n_eval_episodes"],
#     gamma=model_download["gamma"],
#     max_epsilon=model_download["max_epsilon"],
#     min_epsilon=model_download["min_epsilon"],
#     decay_rate=model_download["decay_rate"],
#     eval_seed=model_download["eval_seed"],
# )

# model.train()
# print(model.evaluate_agent())

# model_dict = {
#     "env_id": model.env_id,
#     "max_steps": model.max_steps,
#     "n_training_episodes": model.n_training_episodes,
#     "n_eval_episodes": model.n_eval_episodes,
#     "eval_seed": model.eval_seed,
#     "learning_rate": model.learning_rate,
#     "gamma": model.gamma,
#     "max_epsilon": model.max_epsilon,
#     "min_epsilon": model.min_epsilon,
#     "decay_rate": model.decay_rate,
#     "qtable": model.Qtable,
# }

# username = "JYC333"  # FILL THIS
# repo_name = "q-Taxi-v3"
# push_to_hub(
#     env_id=model.env_id,
#     repo_id=f"{username}/{repo_name}",
#     model=model_dict,
#     env=env,
# )

# print(
#     evaluate_agent(
#         env,
#         model["max_steps"],
#         model["n_eval_episodes"],
#         model["qtable"],
#         model["eval_seed"],
#     )
# )
