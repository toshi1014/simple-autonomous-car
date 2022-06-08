from collections import namedtuple
import numpy as np
from course import Course


MAX_STEERING_ANGLE = np.pi / 6
ACCELERATION_G = 1
DECELERATION_G = 3


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
    def __init__(self, course, initial_position, initial_direction):
        self.course = course
        self.position = initial_position
        self.speed = 0
        self.direction = initial_direction

    def get_steering_angle(self, steering):
        return steering * MAX_STEERING_ANGLE

    def update_position(self, steering):
        steering_angle = self.get_steering_angle(steering)
        self.direction += steering_angle

        self.position["x"] += self.speed * np.cos(self.direction)
        self.position["y"] += self.speed * np.sin(self.direction)

        self.position = self.course.apply_track_limit(self.position)

        return 1

    def update_speed(self, throttle, brake):
        self.speed += throttle * ACCELERATION_G - brake * DECELERATION_G

        return 1

    def lidar_detection(self):
        return [1, 1, 1]


class Environment:
    def __init__(self, course_layout_filepath):
        self.action_space = ActionSpace([
            {"low": -1, "high": 1},     # steering
            {"low": 0, "high": 1},      # throttle
            {"low": 0, "high": 1},      # brake
        ])

        self.course = Course(course_layout_filepath)

        self.initial_position = self.course.initial_position

        self.reset()

    # wall position & speed
    def state_repr(self):
        return np.array([
            self.car_model.position["x"],
            self.car_model.position["y"],
            self.car_model.speed,
        ], dtype=np.float32)
        # wall_position_list = self.car_model.lidar_detection()
        # return np.concatenate(
        #     (wall_position_list, [self.car_model.speed]),
        #     axis=0
        # )

    def reset(self):
        self.position = self.initial_position
        initial_direction = np.pi / 2
        self.car_model = CarModel(
            self.course,
            self.initial_position,
            initial_direction,
        )
        return self.state_repr()
        # return initial state

    def step(self, action):
        steering, throttle, brake = action

        position_reward = self.car_model.update_position(steering)
        pedal_reward = self.car_model.update_speed(throttle, brake)

        reward = position_reward + pedal_reward
        done = False
        info = None

        return self.state_repr(), reward, done, info
        # return next_state, reward, done, info

    def render(self):
        ...


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    speed_list = []
    position_x_list = []
    position_y_list = []

    course_layout_filepath = "course_layout.json"
    env = Environment(course_layout_filepath)

    state = env.reset()

    for i in range(5):
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        position_x_list.append(next_state[0])
        position_y_list.append(next_state[1])
        speed_list.append(next_state[-1])

    ax1.plot(position_x_list, position_y_list)
    ax2.plot(list(range(len(speed_list))), speed_list)
    ax1.set_title("Positions")
    ax2.set_title("Speeds")
    ax1.set_aspect("equal")
    ax2.set_aspect("equal")

    plt.savefig("out.png")
