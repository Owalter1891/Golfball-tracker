class TrajectoryAnalyzer:
    def __init__(self):
        self.points = []
        self.gravity = 0.15
        self.x_speed_loss = 0.025

    def add_point(self, point):
        self.points.append((int(point[0]), int(point[1]), int(point[2])))

    def predict_next_points(self, steps=20):
        """
        Predict the next points based on velocity between last frames.
        Returns list of (x, y)
        """
        if len(self.points) < 2:
            return []

        x1, y1, f1 = self.points[-2]
        x2, y2, f2 = self.points[-1]
        dt = f2 - f1 if f2 != f1 else 1

        vx = (x2 - x1) / dt
        vy = (y2 - y1) / dt

        x, y = x2, y2
        predicted = []

        for frame in range(steps):
            x += vx
            if vx > 0:
                vx -= self.x_speed_loss
            elif vx < 0:
                vx += self.x_speed_loss
            vy += self.gravity
            y += vy
            predicted.append((int(x), int(y), int(f2 + frame*1.75)))

        return predicted