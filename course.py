import json


class Course:
    def __init__(self, course_layout_filepath):
        with open(course_layout_filepath, "r") as f:
            course_layout = json.load(f)
        self.initial_position = course_layout["initial_position"]
        self.goal_area = course_layout["goal_area"]

    def within_range(self, min, target, max):
        return min <= target <= max

    def is_goal(self, position):
        return within_range(
            self.goal_area["x"]["min"],
            position["x"],
            self.goal_area["x"]["max"]
        ) & within_range(
            self.goal_area["y"]["min"],
            position["y"],
            self.goal_area["y"]["max"]
        )

    def apply_track_limit(self, position):
        return position


course = Course("course_layout.json")
