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


class State:
    # speed & wall position
    def __init__(self):
        ...


class Environment:
    def __init__(self, course_layout_filepath):
        self.action_space = ActionSpace([
            {"low": -1, "high": 1},     # sterring
            {"low": 0, "high": 1},      # throttle
            {"low": 0, "high": 1},      # brake
        ])
        self.course = Course(course_layout_filepath)
        self.state = State(self.course.initial_position)

    def reset(self):
        ...
        # return initial state

    def step(self, action):
        ...
        # return next_state, reward, done, info

    def render(self):
        ...


env = Environment()
