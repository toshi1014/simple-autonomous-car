import argparse
import matplotlib.pyplot as plt
from agent import Agent, Trainer
from environment import Environment


COURSE_LAYOUT_FILEPATH = "course_layout.json"
CAR_MODEL_CONFIG_FILEPATH = "config/car_model_config.json"
REWARD_CONFIG_FILEPATH = "config/reward_config.json"

# params
discount_rate = 0.99
actor_learning_rate = 0.001
critic_learning_rate = 0.002
buffer_size = 1024
batch_size = 64
noise_stddev = 0.1
target_trans_rate = 0.005
model_path = "model.h5"
max_episodes = 50
# end params


trainer = Trainer(
    discount_rate=discount_rate,
    buffer_size=buffer_size,
    batch_size=batch_size,
)

agent = Agent(
    actor_learning_rate=actor_learning_rate,
    critic_learning_rate=critic_learning_rate,
    noise_stddev=noise_stddev,
    target_trans_rate=target_trans_rate,
)


def get_args():
    parser = argparse.ArgumentParser(description="Description")

    parser.add_argument(
        "--play", help="play agent", action="store_true",
    )
    parser.add_argument(
        "--test", help="test agent in OpenAI Gym", action="store_true",
    )

    args = parser.parse_args()
    if args.play & args.test:
        raise Exception("No play opt for test agent")

    return args


def main(args):
    env = Environment(
        COURSE_LAYOUT_FILEPATH,
        CAR_MODEL_CONFIG_FILEPATH,
        REWARD_CONFIG_FILEPATH,
    )

    if args.play:
        trained_agent = (agent.__class__).load(model_path)
        trained_agent.play(env, 5)

    else:
        if args.test:
            import gym      # noqa
            # env = gym.make("Pendulum-v1")
            env = gym.make("BipedalWalker-v3")

        trained_agent, reward_hist = trainer.train(
            env, agent, max_episodes
        )
        trained_agent.save(model_path)
        plt.plot(range(len(reward_hist)), reward_hist)
        plt.title("reward_history")
        plt.savefig("reward_history.png")


main(get_args())
