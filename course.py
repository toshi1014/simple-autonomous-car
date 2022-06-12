from collections import namedtuple
import copy
import json
import numpy as np


# ax + by + c = 0
LineEqn = namedtuple("LineEqn", ["a", "b", "c"])


class Course:
    def __init__(self, course_layout_filepath):
        with open(course_layout_filepath, "r") as f:
            self.course_layout_dict = json.load(f)

        self.parse_course_layout()

    # get line written by two points
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

    def get_line_eqn2(self, position, vec):
        if vec in [0, np.pi]:
            return LineEqn(0, 1, -position["y"])
        elif vec in [np.pi/2, np.pi*3/2]:
            return LineEqn(1, 0, -position["x"])
        else:
            a = np.tan(vec)
            c = - a * position["x"] + position["y"]
            return LineEqn(a, -1, c)

    def parse_course_layout(self):
        self.left_wall_list = []
        self.right_wall_list = []

        for str_lf_wall, lf_wall_list in [
            ["left_wall", self.left_wall_list],
            ["right_wall", self.right_wall_list],
        ]:
            wall = self.course_layout_dict["course"][str_lf_wall]
            for i in range(len(wall)-1):
                lf_wall_list.append({
                    "range": [wall[i], wall[i+1]],
                    "eqn": self.get_line_eqn(wall[i], wall[i+1]),
                })

    def within_range(self, min, target, max):
        return min <= target <= max

    def in_area(self, position, area):
        return self.within_range(
            float(area["x"]["min"]),
            position["x"],
            float(area["x"]["max"])
        ) & self.within_range(
            float(area["y"]["min"]),
            position["y"],
            float(area["y"]["max"])
        )

    def is_goal(self, position):
        return self.in_area(position, self.course_layout_dict["goal_area"])

    def does_collide(self, intersection, pre_p, p):
        min_x = min(pre_p["x"], p["x"])
        max_x = max(pre_p["x"], p["x"])

        min_y = min(pre_p["y"], p["y"])
        max_y = max(pre_p["y"], p["y"])

        return self.within_range(
            min_x,
            intersection["x"],
            max_x,
        ) & self.within_range(
            min_y,
            intersection["y"],
            max_y,
        )

    def in_off_limits_area(self, position):
        for off_limits_area in self.course_layout_dict["off_limits_areas"]:
            if self.in_area(position, off_limits_area):
                return True
        return False

    def apply_track_limit(self, pre_position, position):
        if self.in_off_limits_area(position):
            return position, True

        move_eqn = self.get_line_eqn(pre_position, position)

        distance_list = []
        intersection_list = []

        for lf_wall_list in [self.left_wall_list, self.right_wall_list]:
            for wall in lf_wall_list:
                intersection = self.get_intersection(
                    move_eqn, wall["eqn"]
                )

                if intersection:
                    if self.does_collide(
                        intersection, pre_position, position
                    ) & self.in_wall_range(intersection, wall["range"]):
                        distance_list.append(
                            self.get_distance(intersection, pre_position)
                        )
                        intersection_list.append(intersection)

        if len(distance_list) == 0:
            return position, False
        else:
            min_distance_idx = distance_list.index(min(distance_list))
            return intersection_list[min_distance_idx], True

    # check whether intersection is forward
    def is_forward_wall(self, intersection, position, sensor_direction):
        p1 = copy.deepcopy(position)
        p1["x"] += np.cos(sensor_direction)
        p1["y"] += np.sin(sensor_direction)

        p2 = copy.deepcopy(position)
        p2["x"] -= np.cos(sensor_direction)
        p2["y"] -= np.sin(sensor_direction)

        return bool(
            self.get_distance(p1, intersection) <
            self.get_distance(p2, intersection)
        )

    def in_wall_range(self, intersection, wall_range):
        min_x = min(wall_range[0]["x"], wall_range[1]["x"])
        max_x = max(wall_range[0]["x"], wall_range[1]["x"])

        min_y = min(wall_range[0]["y"], wall_range[1]["y"])
        max_y = max(wall_range[0]["y"], wall_range[1]["y"])

        return self.within_range(
            min_x,
            intersection["x"],
            max_x,
        ) & self.within_range(
            min_y,
            intersection["y"],
            max_y,
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
                    if self.is_forward_wall(
                        intersection, position, sensor_direction
                    ) & self.in_wall_range(intersection, wall["range"]):
                        distance_list.append(
                            self.get_distance(intersection, position)
                        )

        if len(distance_list) == 0:
            return float("inf")
        else:
            return min(distance_list)


if __name__ == "__main__":
    course_layout_filepath = "course_layout.json"
    course = Course(course_layout_filepath)
    position = {"x": 0.5, "y": -1}

    print(course.in_off_limits_area(position))
