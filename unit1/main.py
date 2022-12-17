import os
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from gymnasium.utils.play import play
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

from huggingface_sb3 import package_to_hub, load_from_hub


class Unit1:
    def __init__(self, env_id, repo_id, file_name, model_name, n_envs) -> None:
        self.env_id = env_id
        self.repo_id = repo_id
        self.file_name = file_name
        self.model_name = model_name

        self.train_env = make_vec_env("LunarLander-v2", n_envs=n_envs)
        self.model = PPO(
            policy="MlpPolicy",
            env=self.train_env,
            learning_rate=0.0004,
            n_steps=1024,
            batch_size=64,
            n_epochs=4,
            gamma=0.999,
            gae_lambda=0.98,
            ent_coef=0.01,
            verbose=1,
        )

        self.env = gym.make("LunarLander-v2", render_mode="rgb_array")

    def load_model(self, local=True, **kargs):
        if local:
            self.model = PPO.load(
                self.file_name,
                env=self.train_env,
                custom_objects=kargs,
                print_system_info=True,
            )
        else:
            self.model = load_from_hub(self.repo_id, self.model_name)

    def train(self, steps, record=False, record_steps=100000):
        old_model_time = datetime.fromtimestamp(
            os.path.getctime(self.file_name)
        ).strftime("%Y%m%dT%H%M%S")
        os.rename(
            self.file_name,
            self.file_name.split(".")[0] + f"_{old_model_time}.zip",
        )

        if record:
            iter_times = int(steps / record_steps)

            if self.env.__class__.__name__ != "RecordVideo":
                self.env = RecordVideo(
                    self.env,
                    "./record_" + datetime.now().strftime("%Y%m%dT%H%M%S"),
                    episode_trigger=lambda x: x < iter_times,
                )
            else:
                self.env.video_folder = "./record_" + datetime.now().strftime(
                    "%Y%m%dT%H%M%S"
                )
                self.env.episode_trigger = lambda x: x < iter_times

            observation, _ = self.env.reset()

            n = 0
            while n < iter_times:
                # Take a random action
                action = self.model.predict(observation=observation)

                # Do this action in the environment and get
                # next_state, reward, done and info
                observation, _, terminated, truncated, _ = self.env.step(action[0])

                # If the game is done (in our case we land, crashed or timeout)
                if terminated or truncated:
                    # Reset the environment
                    n += 1
                    observation, _ = self.env.reset()
                    if n < iter_times:
                        self.model.learn(total_timesteps=steps)
                        self.model.save(self.model_name)
                        self.evaluation()
        else:
            self.model.learn(total_timesteps=steps)
            self.model.save(self.model_name)
            self.evaluation()

    def evaluation(self):
        mean_reward, std_reward = evaluate_policy(
            self.model, self.train_env, n_eval_episodes=10, deterministic=True
        )
        print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

    def play(self):
        mapping = {"w": 2, "a": 3, "d": 1}
        play(
            gym.make("LunarLander-v2", render_mode="rgb_array"), keys_to_action=mapping
        )

    def push_to_huggingface(self):
        package_to_hub(
            model=self.model,  # Our trained model
            model_name=self.model_name,  # The name of our trained model
            model_architecture="PPO",  # The model architecture we used: in our case PPO
            env_id=self.env_id,  # Name of the environment
            eval_env=make_vec_env("LunarLander-v2", n_envs=1),  # Evaluation Environment
            repo_id=self.repo_id,  # id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name} for instance ThomasSimonini/ppo-LunarLander-v2
            commit_message="Upload PPO LunarLander-v2 trained agent",
        )


unit1 = Unit1(
    env_id="LunarLander-v2",
    repo_id="JYC333/ppo-LunarLander-v2",
    file_name="unit1/ppo-LunarLander-v2.zip",
    model_name="ppo-LunarLander-v2",
    n_envs=1,
)
unit1.play()
