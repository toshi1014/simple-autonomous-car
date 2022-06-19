import argparse
import os
import shutil
import matplotlib.pyplot as plt
from agent import Agent, Trainer
from environment import Environment


COURSE_LAYOUT_FILEPATH = os.path.join("course_layouts", "course_layout2.json")
CAR_MODEL_CONFIG_FILEPATH = os.path.join("config", "car_model_config.json")
REWARD_CONFIG_FILEPATH = os.path.join("config", "reward_config.json")
LOG_DIR = "logs"


# params
discount_rate = 0.99
actor_learning_rate = 0.001
critic_learning_rate = 0.002
buffer_size = 1024
batch_size = 64
noise_stddev = 0.1
target_trans_rate = 0.005
model_path = "model.h5"
max_episodes = 500
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
    parser.add_argument(
        "--log", help="save logs", action="store_true",
    )

    args = parser.parse_args()
    if args.play & args.test:
        raise Exception("No play opt for test agent")

    return args


def main(args):
    global LOG_DIR

    if args.log:
        if os.path.exists(LOG_DIR):
            shutil.rmtree(LOG_DIR)

        os.mkdir(LOG_DIR)
        os.mkdir(os.path.join(LOG_DIR, "bin"))
        os.mkdir(os.path.join(LOG_DIR, "img"))
    else:
        LOG_DIR = None

    env = Environment(
        COURSE_LAYOUT_FILEPATH,
        CAR_MODEL_CONFIG_FILEPATH,
        REWARD_CONFIG_FILEPATH,
    )

    if args.play:
        trained_agent = (agent.__class__).load(model_path)
        trained_agent.play(env, LOG_DIR)

    else:
        if args.test:
            import gym      # noqa
            env = gym.make("Pendulum-v1")
            # env = gym.make("BipedalWalker-v3")

        trained_agent, reward_hist = trainer.train(
            env, agent, max_episodes, LOG_DIR
        )
        trained_agent.save(model_path)
        plt.plot(range(len(reward_hist)), reward_hist)
        plt.title("reward_history")
        plt.savefig("reward_history.png")


main(get_args())
