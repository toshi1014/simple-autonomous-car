from collections import namedtuple
import json
import numpy as np


LineEqn = namedtuple("LineEqn", ["a", "b", "c"])


class Course:
    def __init__(self, course_layout_filepath):
        with open(course_layout_filepath, "r") as f:
            course_layout_dict = json.load(f)

        self.initial_position = course_layout_dict["initial_position"]
        self.goal_area = course_layout_dict["goal_area"]
        self.parse_course_layout(course_layout_dict)

    # get line written by p1 & p2
    def get_line_eqn(self, p1, p2):
        vec_x = p2["x"] - p1["x"]
        vec_y = p2["y"] - p1["y"]

        if vec_x == 0:
            return LineEqn(1, 0, -p1["x"])
        elif vec_y == 0:
            return LineEqn(0, 1, -p1["y"])
        else:
            a = vec_y / vec_x
            c = - a * p1["x"] + p1["y"]
            return LineEqn(a, -1, c)

    def get_line_eqn2(self, position, sensor_direction):
        if sensor_direction in [0, np.pi]:
            return LineEqn(0, 1, -position["y"])
        elif sensor_direction in [np.pi/2, np.pi*3/4]:
            return LineEqn(1, 0, -position["x"])
        else:
            a = np.tan(sensor_direction)
            c = - a * position["x"] + position["y"]
            return LineEqn(a, -1, c)

    def parse_course_layout(self, course_layout_dict):
        self.left_wall_list = []
        self.right_wall_list = []

        for str_lf_wall, lf_wall_list in [
            ["left_wall", self.left_wall_list],
            ["right_wall", self.right_wall_list],
        ]:
            wall = course_layout_dict["course"][str_lf_wall]
            for i in range(len(wall)-1):
                lf_wall_list.append({
                    "range": [wall[i], wall[i+1]],
                    "eqn": self.get_line_eqn(wall[i], wall[i+1]),
                })

    def within_range(self, min, target, max):
        return min <= target <= max

    def is_goal(self, position):
        return self.within_range(
            self.goal_area["x"]["min"],
            position["x"],
            self.goal_area["x"]["max"]
        ) & self.within_range(
            self.goal_area["y"]["min"],
            position["y"],
            self.goal_area["y"]["max"]
        )

    def apply_track_limit(self, pre_position, position):
        return position

    def in_wall_range(self, intersection, wall_range):
        return self.within_range(
            wall_range[0]["x"],
            intersection["x"],
            wall_range[1]["x"],
        ) & self.within_range(
            wall_range[0]["y"],
            intersection["y"],
            wall_range[1]["y"],
        )

    def get_intersection(self, line1, line2):
        if (line1.a * line2.b) == (line1.b * line2.a) == 0:   # parallel
            return None
        else:
            return {
                "x": (line1.b*line2.c - line2.b*line1.c)
                / (line1.a*line2.b - line2.a*line1.b),
                "y": (line2.a*line1.c - line1.a*line2.c)
                / (line1.a*line2.b - line2.a*line1.b),
            }

    def get_distance(self, p1, p2):
        return np.sqrt((p1["x"] - p2["x"])**2 + (p1["y"] - p2["y"])**2)

    def get_wall_distance(self, position, sensor_direction):
        sensor_line_eqn = self.get_line_eqn2(position, sensor_direction)

        distance_list = []

        for lf_wall_list in [self.left_wall_list, self.right_wall_list]:
            for wall in lf_wall_list:
                intersection = self.get_intersection(
                    sensor_line_eqn, wall["eqn"]
                )

                if intersection:
                    if self.in_wall_range(intersection, wall["range"]):
                        distance_list.append(
                            self.get_distance(intersection, position)
                        )

        if len(distance_list) == 0:
            return float("inf")
        else:
            return min(distance_list)
