from collections import deque
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from base import Agent, Trainer


class DDPG(Agent):
    def __init__(
        self, actor_learning_rate, critic_learning_rate,
        noise_stddev=None, target_trans_rate=None,
    ):
        super().__init__()
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.noise_stddev = noise_stddev
        self.target_trans_rate = target_trans_rate

    def save(self, model_path):
        self.actor.save(model_path, overwrite=True, include_optimizer=False)

    @classmethod
    def load(cls, model_path):
        agent = cls(0, 0, 0)
        agent.actor = keras.models.load_model(model_path)
        agent.action_space = list(range(agent.actor.output.shape[1]))
        agent.model = True
        return agent

    def make_model(self, feature_shape):
        # actor model
        input = keras.Input(shape=(feature_shape,))
        dense1 = keras.layers.Dense(
            units=512,
            activation=tf.nn.relu,
        )(input)
        dense2 = keras.layers.Dense(
            units=512,
            activation=tf.nn.relu,
        )(dense1)
        output = keras.layers.Dense(
            units=self.action_space[0],
            activation=tf.nn.tanh,
        )(dense2)

        self.actor = keras.Model(inputs=input, outputs=output)
        self.actor.compile(optimizer=keras.optimizers.Adam(
            learning_rate=self.actor_learning_rate
        ))
        self.target_actor = keras.models.clone_model(self.actor)
        # end actor model

        # critic model
        input_state = keras.Input(shape=(feature_shape,))
        input_action = keras.Input(shape=(self.action_space[0],))
        input = tf.concat([input_state, input_action], axis=1)
        dense1 = keras.layers.Dense(
            units=512,
            activation=tf.nn.relu,
        )(input)
        dense2 = keras.layers.Dense(
            units=512,
            activation=tf.nn.relu,
        )(dense1)
        output = keras.layers.Dense(units=1)(dense2)
        self.critic = keras.Model(
            inputs=[input_state, input_action],
            outputs=output
        )
        self.critic.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=self.critic_learning_rate
            ),
            loss=tf.losses.mean_squared_error,
        )
        self.target_critic = keras.models.clone_model(self.critic)
        # end critic model

        self.model = True

    def update(self, experience_list, discount_rate):
        state_list = np.array([e.state for e in experience_list])
        action_list = np.array([e.action for e in experience_list])
        reward_list = np.array([[e.reward] for e in experience_list])
        next_state_list = np.array([e.next_state for e in experience_list])
        int_done_list = np.array([[int(e.done)] for e in experience_list])

        # update actor
        with tf.GradientTape() as tape:
            new_action_list = self.actor(state_list, training=False)
            actor_loss = - tf.math.reduce_mean(
                self.critic([state_list, new_action_list])
            )

        grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(
            zip(grad, self.actor.trainable_variables)
        )
        # end update actor

        # update critic
        next_action_list = self.target_actor(state_list, training=False)
        discounted_reward_list = reward_list + self.target_critic(
            [next_state_list, next_action_list]
        ) * discount_rate * (1 - int_done_list)

        self.critic.train_on_batch(
            [state_list, action_list],
            discounted_reward_list
        )
        # end update critic

        self.update_target_model()

    def update_target_model(self):

        def target_trans_func(w, target_w):
            return w * self.target_trans_rate + \
                target_w * (1 - self.target_trans_rate)

        new_target_actor_weight = [
            target_trans_func(w, target_w)
            for w, target_w in zip(
                self.actor.weights, self.target_actor.weights
            )
        ]
        new_target_critic_weight = [
            target_trans_func(w, target_w)
            for w, target_w in zip(
                self.critic.weights, self.target_critic.weights
            )
        ]

        self.target_actor.set_weights(new_target_actor_weight)
        self.target_critic.set_weights(new_target_critic_weight)


class DDPGTrainer(Trainer):
    def __init__(self, discount_rate, buffer_size, batch_size):
        super().__init__(discount_rate)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.experience_list = deque(maxlen=self.buffer_size)

    def train(self, env, agent, max_episodes):
        agent.min_action = env.action_space.low
        agent.max_action = env.action_space.high
        agent.action_space = list(env.action_space.shape)
        agent.random_action = env.action_space.sample
        reward_hist = super().train(env, agent, max_episodes)
        return agent, reward_hist

    def step(self, agent, experience):
        if not agent.model:
            if len(self.experience_list) == self.buffer_size:
                agent.make_model(self.experience_list[0].state.shape[0])

        else:
            batch = random.sample(self.experience_list, self.batch_size)
            agent.update(batch, self.discount_rate)
