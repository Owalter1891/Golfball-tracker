import numpy as np

class Trajectory:
    """A simple class to store the trajectory history of an object."""
    def __init__(self):
        self.points = []

    def add_point(self, point):
        """
        Adds a new point to the trajectory history.
        Point should be a tuple (x, y).
        """
        self.points.append(point)

    def get_points_array(self):
        """
        Returns the trajectory points as a NumPy array for drawing.
        """
        return np.array(self.points, dtype=np.int32)

    def __len__(self):
        return len(self.points)