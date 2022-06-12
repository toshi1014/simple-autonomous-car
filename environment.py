import copy
import json
import matplotlib.pyplot as plt
import numpy as np
from course import Course


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
    def __init__(self, course, config_filepath):
        with open(config_filepath, "r") as f:
            self.car_model_config = json.load(f)

        self.course = course
        self.position = self.course.course_layout_dict["initial_position"]
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
        return steering  # * self.max_steering_angle

    # if steering > 0, then turn left
    def update_position(self, steering):
        if self.speed != 0:
            steering_angle = self.get_steering_angle(steering)
            self.car_direction += steering_angle

            pre_position = copy.deepcopy(self.position)

            self.position["x"] += self.speed * np.cos(self.car_direction)
            self.position["y"] += self.speed * np.sin(self.car_direction)

            self.position, bool_off_limits = self.course.apply_track_limit(
                pre_position, self.position
            )

            if bool_off_limits:
                return False

        # TODO: add reward
            return 1
        else:
            return 1

    def update_speed(self, throttle, brake):
        self.speed += throttle * self.car_model_config["acceleration_g"] \
            - brake * self.car_model_config["deceleration_g"]

        self.speed = max(self.speed, 0)

        # TODO: add simul throttle & brake penalty

        return 1

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
    def __init__(self, course_layout_filepath):
        self.action_space = ActionSpace([
            {"low": -1, "high": 1},     # steering
            {"low": 0, "high": 1},      # throttle
            {"low": 0, "high": 1},      # brake
        ])

        self.course = Course(course_layout_filepath)

        self.reset()

        # add history
        self.position_hist = [copy.deepcopy(self.car_model.position)]
        self.speed_hist = [self.car_model.speed]
        self.steering_hist = []
        self.throttle_hist = []
        self.brake_hist = []

    # wall position & speed
    def state_repr(self):
        return np.concatenate(
            (self.car_model.lidar_detection(), [self.car_model.speed]),
            axis=0,
        )

    def reset(self):
        self.car_model = CarModel(self.course, "car_model_config.json")
        return self.state_repr()

    def step(self, action):
        steering, throttle, brake = action

        pedal_reward = self.car_model.update_speed(throttle, brake)
        position_reward = self.car_model.update_position(steering)

        reward = position_reward + pedal_reward
        done = not bool(position_reward)
        info = None

        # add history
        self.position_hist.append(copy.deepcopy(self.car_model.position))
        self.speed_hist.append(self.car_model.speed)
        self.steering_hist.append(steering)
        self.throttle_hist.append(throttle)
        self.brake_hist.append(brake)

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

    def render(self):
        fig = plt.figure()

        # position history
        ax1 = fig.add_subplot(2, 2, 1)
        ax1 = self.draw_course(ax1)
        x_list = [p["x"] for p in self.position_hist]
        y_list = [p["y"] for p in self.position_hist]
        ax1.plot(x_list, y_list, marker="o")
        ax1.set_title("Position")
        ax1.set_aspect("equal")

        # speed history
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(list(range(len(self.speed_hist))), self.speed_hist)
        ax2.set_title("Speed")

        # steering history
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(list(range(len(self.steering_hist))), self.steering_hist)
        ax3.hlines(0, 0, len(self.steering_hist), color="k", linestyle="--")
        ax3.hlines(1, 0, len(self.steering_hist), color="k", linestyle="-")
        ax3.hlines(-1, 0, len(self.steering_hist), color="k", linestyle="-")
        ax3.set_title("Steering")

        # throttle & brake history
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.plot(
            list(range(len(self.throttle_hist))), self.throttle_hist,
            label="Throttle", color="red"
        )
        ax4.plot(
            list(range(len(self.brake_hist))), self.brake_hist,
            label="Brake", color="blue"
        )
        ax4.hlines(0, 0, len(self.steering_hist), color="k", linestyle="-")
        ax4.hlines(1, 0, len(self.steering_hist), color="k", linestyle="-")
        ax4.set_title("Throttle & Brake")
        ax4.legend(loc="best", prop={"size": 6})

        plt.subplots_adjust(hspace=0.3)
        plt.savefig("out.png", bbox_inches="tight", pad_inches=0.1)


if __name__ == '__main__':
    course_layout_filepath = "course_layout.json"
    env = Environment(course_layout_filepath)

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
    env.render()
