import copy
import json
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from .course import Course


class ActionSpace:
    def __init__(self, bound_list):
        self.low = np.array(
            [bound["low"] for bound in bound_list], dtype=np.float32
        )
        self.high = np.array(
            [bound["high"] for bound in bound_list], dtype=np.float32
        )
        self.shape = (len(bound_list),)

    def sample(self):
        return np.array(
            [
                np.random.uniform(self.low[i], self.high[i])
                for i in range(self.shape[0])
            ],
            dtype=np.float32
        )


class CarModel:
    def __init__(self, course, car_model_config_filepath):
        with open(car_model_config_filepath, "r") as f:
            self.car_model_config = json.load(f)

        self.course = course
        self.position = copy.deepcopy(
            self.course.course_layout_dict["initial_position"]
        )
        self.car_direction = np.deg2rad(
            self.course.course_layout_dict["initial_direction"]
        )
        self.max_steering_angle = np.deg2rad(
            self.car_model_config["max_steering_angle"]
        )

        self.speed = 0      # TODO: random initial speed
        self.sensor_direction_list = [
            np.deg2rad(sd)
            for sd in self.car_model_config["sensor_directions"]
        ]

    def get_steering_angle(self, steering):
        return steering * self.max_steering_angle

    # if steering > 0, then turn left
    def update_position(self, steering):
        if self.speed != 0:
            steering_angle = self.get_steering_angle(steering)
            self.car_direction += steering_angle

            pre_position = copy.deepcopy(self.position)

            self.position["x"] += self.speed * np.cos(self.car_direction)
            self.position["y"] += self.speed * np.sin(self.car_direction)

            self.position, bool_off_limits, bool_goal = \
                self.course.apply_track_limits(
                    pre_position, self.position
                )

            return bool_off_limits, bool_goal

        else:
            return False, False

    def update_speed(self, throttle, brake):
        self.speed += throttle * self.car_model_config["acceleration_g"] \
            - brake * self.car_model_config["deceleration_g"]

        self.speed = max(self.speed, 0)

        return bool((throttle * brake) > 0)

    def lidar_detection(self):
        wall_distance_list = []

        for sensor_direction in self.sensor_direction_list:
            sensor_direction_now = self.car_direction + sensor_direction
            wall_distance = self.course.get_wall_distance(
                self.position,
                sensor_direction_now,
            )
            wall_distance_list.append(
                min(wall_distance, self.car_model_config["lidar_range"])
            )

        return wall_distance_list


class Environment:
    def __init__(
        self, course_layout_filepath,
        car_model_config_filepath, reward_config_filepath,
    ):
        self.car_model_config_filepath = car_model_config_filepath
        self.course = Course(course_layout_filepath)

        with open(reward_config_filepath, "r") as f:
            self.reward_config = json.load(f)

        self.action_space = ActionSpace([
            {"low": -1, "high": 1},     # steering
            {"low": 0, "high": 1},      # throttle
            {"low": 0, "high": 1},      # brake
        ])

        self.reset()

    # wall position & speed
    def state_repr(self):
        return np.concatenate(
            (self.car_model.lidar_detection(), [self.car_model.speed]),
            axis=0,
        )

    def reset(self):
        self.car_model = CarModel(self.course, self.car_model_config_filepath)

        # logs
        self.position_log = [copy.deepcopy(self.car_model.position)]
        self.speed_log = [self.car_model.speed]
        self.steering_log = []
        self.throttle_log = []
        self.brake_log = []

        return self.state_repr()

    def step(self, action):
        steering, throttle, brake = action

        bool_throttle_with_brake = self.car_model.update_speed(throttle, brake)
        bool_off_limits, bool_goal = self.car_model.update_position(steering)

        if bool_goal:
            reward = self.reward_config["goal_reward"]
        else:
            if bool_off_limits:
                reward = self.reward_config["off_limits_penalty"]
            elif bool_throttle_with_brake:
                reward = self.reward_config["throttle_with_brake_penalty"]
            else:
                reward = self.reward_config["default_reward"]

        done = bool_off_limits | bool_goal
        info = None

        # add into logs
        self.position_log.append(copy.deepcopy(self.car_model.position))
        self.speed_log.append(self.car_model.speed)
        self.steering_log.append(steering)
        self.throttle_log.append(throttle)
        self.brake_log.append(brake)

        return self.state_repr(), reward, done, info

    def draw_course(self, ax):
        left_wall_xy_list = np.array([
            [w["x"], w["y"]]
            for w in self.course.course_layout_dict["course"]["left_wall"]
        ])
        right_wall_xy_list = np.array([
            [w["x"], w["y"]]
            for w in self.course.course_layout_dict["course"]["right_wall"]
        ])

        ax.plot(left_wall_xy_list[:, 0], left_wall_xy_list[:, 1], color="k")
        ax.plot(right_wall_xy_list[:, 0], right_wall_xy_list[:, 1], color="k")
        return ax

    def save_log(self, dir, filename):
        # save logs
        filepath = os.path.join(dir, "bin", filename + ".pickle")
        with open(filepath, "wb") as f:
            obj = {
                "position_log": self.position_log,
                "speed_log": self.speed_log,
                "steering_log": self.steering_log,
                "throttle_log": self.throttle_log,
                "brake_log": self.brake_log,
            }
            pickle.dump(obj, f)

        fig = plt.figure()

        # position log
        ax1 = fig.add_subplot(2, 2, 1)
        ax1 = self.draw_course(ax1)
        x_list = [p["x"] for p in self.position_log]
        y_list = [p["y"] for p in self.position_log]
        ax1.plot(x_list, y_list, marker="o")
        ax1.set_title("Position")
        ax1.set_aspect("equal")

        # speed log
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(list(range(len(self.speed_log))), self.speed_log)
        ax2.set_title("Speed")

        # steering log
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(list(range(len(self.steering_log))), self.steering_log)
        ax3.hlines(0, 0, len(self.steering_log), color="k", linestyle="--")
        ax3.hlines(1, 0, len(self.steering_log), color="k", linestyle="-")
        ax3.hlines(-1, 0, len(self.steering_log), color="k", linestyle="-")
        ax3.set_title("Steering")

        # throttle & brake log
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.plot(
            list(range(len(self.throttle_log))), self.throttle_log,
            label="Throttle", color="red"
        )
        ax4.plot(
            list(range(len(self.brake_log))), self.brake_log,
            label="Brake", color="blue"
        )
        ax4.hlines(0, 0, len(self.steering_log), color="k", linestyle="-")
        ax4.hlines(1, 0, len(self.steering_log), color="k", linestyle="-")
        ax4.set_title("Throttle & Brake")
        ax4.legend(loc="best", prop={"size": 6})

        plt.subplots_adjust(hspace=0.3)

        filepath = os.path.join(dir, "img", filename + ".png")
        plt.savefig(filepath, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)


if __name__ == '__main__':
    course_layout_filepath = "course_layout.json"
    config_dir = "config"
    car_model_config_filepath = "car_model_config.json"
    reward_config_filepath = "reward_config.json"
    env = Environment(
        course_layout_filepath,
        os.path.join(config_dir, car_model_config_filepath),
        os.path.join(config_dir, reward_config_filepath),
    )

    state = env.reset()

    speed_list = [env.car_model.speed]
    position_x_list = [env.car_model.position["x"]]
    position_y_list = [env.car_model.position["y"]]

    while True:
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(np.array(action))
        print(action)
        position_x_list.append(env.car_model.position["x"])
        position_y_list.append(env.car_model.position["y"])
        speed_list.append(next_state[-1])

        if done:
            print(env.car_model.position)
            break
    env.save_log("logs")
