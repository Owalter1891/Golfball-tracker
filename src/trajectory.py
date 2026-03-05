import numpy as np

class TrajectoryAnalyzer:
    def __init__(self, min_points_for_fit=5):
        self.points = []
        self.max_x_jump = 15
        self.min_points_for_fit = min_points_for_fit
        self.coeffs = None

    def add_point(self, point):
        """
        Add point only if it does not jump too far in X.
        Then, attempt to fit a new trajectory curve.
        point = (x, y, frame_number)
        """
        x, y, f = int(point[0]), int(point[1]), int(point[2])

        if len(self.points) > 0:
            last_x = self.points[-1][0]
            if abs(x - last_x) > self.max_x_jump:
                return

        self.points.append((x, y, f))
        self._fit_trajectory()

    def _fit_trajectory(self):
        """
        Fit a parabolic curve (2nd degree polynomial) to the observed points.
        The result is stored in self.coeffs.
        """
        if len(self.points) < self.min_points_for_fit:
            self.coeffs = None
            return

        x_coords = np.array([p[0] for p in self.points])
        y_coords = np.array([p[1] for p in self.points])

        if (np.max(x_coords) - np.min(x_coords)) < 10:
            self.coeffs = None
            return

        try:
            self.coeffs = np.polyfit(x_coords, y_coords, 2)
        except (np.linalg.LinAlgError, TypeError):
            self.coeffs = None

    def predict_next_points(self, steps=50):
        """
        Predict future points along the fitted parabolic trajectory.
        Returns a list of predicted (x, y, frame_number) tuples.
        """
        if self.coeffs is None or len(self.points) < 2:
            return []

        num_points_for_vx = min(len(self.points), 5)
        recent_points = self.points[-num_points_for_vx:]
        dx = recent_points[-1][0] - recent_points[0][0]
        dt = recent_points[-1][2] - recent_points[0][2]

        if dt == 0 or abs(dx) < 1:
            return []
        vx = dx / dt

        poly_fn = np.poly1d(self.coeffs)
        last_x, _, last_frame = self.points[-1]

        predicted = []
        for i in range(1, steps + 1):
            next_x = last_x + vx * i
            next_y = poly_fn(next_x)
            predicted.append((int(next_x), int(next_y), last_frame + i))
        return predicted