import numpy as np
import random
import imageio
from tqdm import tqdm
from PIL import Image, ImageFont, ImageDraw


class QLearning:
    def __init__(
        self,
        env,
        n_training_episodes=10000,
        n_evaluations=2,
        learning_rate=0.7,
        n_eval_episodes=100,
        max_steps=99,
        gamma=0.95,
        eval_seed=[],
        max_epsilon=1.0,
        min_epsilon=0.05,
        decay_rate=0.0005,
    ) -> None:
        self.env = env
        self.env_id = env.spec.id
        self.Qtable = np.zeros((env.observation_space.n, env.action_space.n))

        self.n_training_episodes = n_training_episodes
        self.n_evaluations = n_evaluations
        self.eval_freq = int(n_training_episodes / n_evaluations)
        self.learning_rate = learning_rate
        self.n_eval_episodes = n_eval_episodes
        self.max_steps = max_steps
        self.eval_seed = eval_seed
        self.gamma = gamma
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate

    def epsilon_greedy_policy(self, state, epsilon):
        # Randomly generate a number between 0 and 1
        random_num = random.uniform(0, 1)
        # if random_num > greater than epsilon --> exploitation
        if random_num > epsilon:
            # Take the action with the highest value given a state (greedy_policy)
            action = np.argmax(self.Qtable[state])
        # else --> exploration
        else:
            action = self.env.action_space.sample()  # Take a random action

        return action

    def evaluate_agent(self, seed=[]):
        """
        Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
        :param env: The evaluation environment
        :param n_eval_episodes: Number of episode to evaluate the agent
        :param Q: The Q-table
        :param seed: The evaluation seed array (for taxi-v3)
        """
        episode_rewards = []
        for episode in range(self.n_eval_episodes):
            if seed:
                state = self.env.reset(seed=seed[episode])[0]
            elif self.eval_seed:
                state = self.env.reset(seed=self.eval_seed[episode])[0]
            else:
                state = self.env.reset()[0]
            step = 0
            total_rewards_ep = 0

            for step in range(self.max_steps):
                # Take the action (index) that have the maximum expected future reward given that state
                action = np.argmax(self.Qtable[state])
                observation, reward, terminated, truncated, info = self.env.step(action)
                total_rewards_ep += reward

                if terminated:
                    break
                state = observation
            episode_rewards.append(total_rewards_ep)
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)

        return mean_reward, std_reward

    def train(self, trial=None):
        eval_step = 0
        for episode in tqdm(range(self.n_training_episodes)):
            # Reduce epsilon (because we need less and less exploration)
            epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(
                -self.decay_rate * episode
            )
            # Reset the environment
            state = self.env.reset()[0]
            step = 0

            # repeat
            for step in range(self.max_steps):
                # Choose the action At using epsilon greedy policy
                action = self.epsilon_greedy_policy(state, epsilon)

                # Take action At and observe Rt+1 and St+1
                # Take the action (a) and observe the outcome state(s') and reward (r)
                observation, reward, terminated, truncated, info = self.env.step(action)

                # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
                self.Qtable[state][action] += self.learning_rate * (
                    reward
                    + self.gamma * np.max(self.Qtable[observation])
                    - self.Qtable[state][action]
                )

                # If done, finish the episode
                if terminated:
                    break

                # Our next state is the new state
                state = observation

            if trial and self.eval_freq > 0 and episode % self.eval_freq == 0:
                eval_step += 1
                mean_reward, std_reward = self.evaluate_agent()
                trial.report(mean_reward - std_reward, eval_step)
                if trial.should_prune():
                    return False
        return True

    def train_with_video(self, out_directory, fps, trial=None):
        images = []
        video_cut = 0

        eval_step = 0
        for episode in tqdm(range(self.n_training_episodes)):
            # Reduce epsilon (because we need less and less exploration)
            epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(
                -self.decay_rate * episode
            )
            # Reset the environment
            state = self.env.reset()[0]
            step = 0
            img = self.env.render()
            img = Image.fromarray(img)
            img = ImageDraw.Draw(img)
            img.text((220, 10), str(episode), fill="black")
            images.append(np.asarray(img._image))

            # repeat
            for step in range(self.max_steps):
                # Choose the action At using epsilon greedy policy
                action = self.epsilon_greedy_policy(state, epsilon)

                # Take action At and observe Rt+1 and St+1
                # Take the action (a) and observe the outcome state(s') and reward (r)
                observation, reward, terminated, truncated, info = self.env.step(action)

                img = self.env.render()
                img = Image.fromarray(img)
                img = ImageDraw.Draw(img)
                img.text((220, 10), str(episode), fill="black")
                images.append(np.asarray(img._image))

                # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
                self.Qtable[state][action] += self.learning_rate * (
                    reward
                    + self.gamma * np.max(self.Qtable[observation])
                    - self.Qtable[state][action]
                )

                # If done, finish the episode
                if terminated:
                    break

                # Our next state is the new state
                state = observation

            if episode % 1000 == 0:
                imageio.mimsave(
                    out_directory / f"train_{video_cut}.mp4",
                    [np.array(img) for i, img in enumerate(images)],
                    fps=fps,
                )
                video_cut += 1
                images = []

            if trial and self.eval_freq > 0 and episode % self.eval_freq == 0:
                eval_step += 1
                mean_reward, std_reward = self.evaluate_agent()
                trial.report(mean_reward - std_reward, eval_step)
                if trial.should_prune():
                    return False

        imageio.mimsave(
            out_directory / f"train_{video_cut}.mp4",
            [np.array(img) for i, img in enumerate(images)],
            fps=fps,
        )

        return True
