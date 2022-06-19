from collections import namedtuple
import numpy as np
import tensorflow as tf
from tensorflow import keras


Experience = namedtuple(
    "Experience",
    ["state", "action", "reward", "next_state", "done"]
)


class Agent:
    def __init__(self):
        self.model = None

    def save(self, model_path):
        self.model.save(model_path, overwrite=True, include_optimizer=False)

    @classmethod
    def load(cls, model_path):
        agent = cls(0.)
        agent.model = keras.models.load_model(model_path)
        agent.action_space = list(range(agent.model.output.shape[1]))
        return agent

    def policy(self, state, greedy=False):
        if self.model:
            action = self.actor(np.array([state]), training=False)[0]

            # NOTE: unique to each action range
            if not greedy:
                steering_noise = tf.random.normal(
                    shape=((1,)),
                    mean=0.0,
                    stddev=self.noise_stddev,
                )
                throttle_noise = tf.random.normal(
                    shape=((1,)),
                    mean=0.0,
                    stddev=0.3,
                )
                brake_noise = tf.random.normal(
                    shape=((1,)),
                    mean=0.0,
                    stddev=self.noise_stddev,
                )
                action += tf.concat([
                    steering_noise,
                    throttle_noise,
                    brake_noise,
                ], 0)

            clipped = tf.clip_by_value(
                action, self.min_action, self.max_action
            )
            # print(clipped.numpy())
            return clipped.numpy()
        else:
            return self.random_action()

    def play(self, env, log_dir):
        self.min_action = env.action_space.low
        self.max_action = env.action_space.high

        state = env.reset()
        done = False
        sum_reward = 0

        while not done:
            action = self.policy(state, greedy=True)
            next_state, reward, done, info = env.step(action)
            sum_reward += reward
            state = next_state

        if log_dir:
            env.save_log(log_dir, "play")

        print(f"reward: {sum_reward}")


class Trainer:
    def __init__(self, discount_rate):
        self.discount_rate = discount_rate

    def step(self, agent, experience):
        ...

    def episode_end(self, agent, episode):
        ...

    def train(self, env, agent, max_episodes, log_dir):
        reward_hist = []

        for episode in range(max_episodes):
            state = env.reset()
            done = False
            sum_reward = 0

            while not done:
                action = agent.policy(state)
                next_state, reward, done, info = env.step(action)
                experience = Experience(
                    state, action, reward, next_state, done
                )
                self.experience_list.append(experience)
                self.step(agent, experience)

                state = next_state
                sum_reward += reward

            self.episode_end(agent, episode)
            reward_hist.append(sum_reward)

            if (log_dir is not None) & bool(agent.model):
                env.save_log(log_dir, f"Episode{episode}")

            if episode % 10 == 0:
                print("Episode", episode)

        return reward_hist
