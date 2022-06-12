import copy
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
        # FIXME: car model params
        self.course = course
        self.position = initial_position
        self.speed = 0
        self.direction = initial_direction
        self.sensor_direction_list = [-np.pi/4, 0, np.pi/4]
        self.lidar_range = 3
        self.car_direction = np.pi/2

    def get_steering_angle(self, steering):
        # TEMP: steering angle
        return steering  # * MAX_STEERING_ANGLE

    def get_car_direction(self, pre_position, position):
        vec_x = position["x"] - pre_position["x"]
        vec_y = position["y"] - pre_position["y"]

        if vec_x == 0:
            if vec_y > 0:
                return np.pi / 2
            else:
                return np.pi * 3 / 2
        else:
            return np.arctan(vec_y / vec_x)

    # if steering > 0, then turn left
    def update_position(self, steering):
        steering_angle = self.get_steering_angle(steering)
        self.direction += steering_angle

        pre_position = copy.deepcopy(self.position)

        self.position["x"] += self.speed * np.cos(self.direction)
        self.position["y"] += self.speed * np.sin(self.direction)

        _, bool_off_limits = self.course.apply_track_limit(
            pre_position, self.position
        )

        if bool_off_limits:
            return False

        self.car_direction = self.get_car_direction(
            pre_position, self.position
        )

        return 1

    def update_speed(self, throttle, brake):
        self.speed += throttle * ACCELERATION_G - brake * DECELERATION_G

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
            wall_distance_list.append(min(wall_distance, self.lidar_range))

        return wall_distance_list


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
        return np.concatenate(
            (self.car_model.lidar_detection(), [self.car_model.speed]),
            axis=0,
        )

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

        pedal_reward = self.car_model.update_speed(throttle, brake)
        position_reward = self.car_model.update_position(steering)

        reward = position_reward + pedal_reward
        done = not bool(position_reward)
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

    course_layout_filepath = "course_layout.json"
    env = Environment(course_layout_filepath)

    state = env.reset()

    speed_list = [env.car_model.speed]
    position_x_list = [env.car_model.position["x"]]
    position_y_list = [env.car_model.position["y"]]

    action_list = [
        [0, 1, 0],
        [-np.pi/4, 0, 0],
        # [-np.pi/3, 1, 0],
        [-np.pi/4, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
    ]

    for action in action_list:
        # action = env.action_space.sample()
        next_state, reward, done, info = env.step(np.array(action))
        print(env.car_model.position)
        print(next_state, "\n")
        position_x_list.append(env.car_model.position["x"])
        position_y_list.append(env.car_model.position["y"])
        speed_list.append(next_state[-1])

        if done:
            break

    ax1.plot(position_x_list, position_y_list, marker="o")
    ax2.plot(list(range(len(speed_list))), speed_list, marker="o")
    ax1.set_title("Positions")
    ax2.set_title("Speeds")
    ax1.set_aspect("equal")
    ax2.set_aspect("equal")

    plt.savefig("out.png")
