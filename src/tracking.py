import cv2
import numpy as np

class KalmanTracker:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)
        
        self.drag = 0.993

        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, self.drag, 0],
            [0, 0, 0, self.drag]
        ], np.float32)
        
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        
        self.first_detection = True
        
        self.gravity = 0.001

    def update(self, measurement):
        """
        Update the filter with a new detection (x, y).
        """
        x, y = measurement
        if self.first_detection:
            self.kf.statePre = np.array([[x], [y], [0], [0]], np.float32)
            self.kf.statePost = np.array([[x], [y], [0], [0]], np.float32)
            self.first_detection = False
            
        meas = np.array([[np.float32(x)], [np.float32(y)]])
        self.kf.correct(meas)
        
    def predict(self):
        """
        Predict the next state.
        Returns (x, y) prediction.
        """
        if not self.first_detection:
            self.kf.statePost[3, 0] += self.gravity

        prediction = self.kf.predict()
        prediction = prediction.flatten()
        return int(prediction[0]), int(prediction[1])

    def get_state(self):
        return int(self.kf.statePost[0, 0]), int(self.kf.statePost[1, 0])

    def predict_future(self, steps):
        """
        Predict future trajectory for a number of steps.
        This runs prediction on a copy of the state to not mess up the main filter's state.
        Returns a list of (x, y) points.
        """
        if self.first_detection:
            return []

        state = self.kf.statePost.copy()
        
        future_points = []
        for _ in range(steps):
            state = np.dot(self.kf.transitionMatrix, state)
            future_points.append((int(state[0, 0]), int(state[1, 0])))
        
        return future_points