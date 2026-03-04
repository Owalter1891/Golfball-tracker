import cv2
import numpy as np

class KalmanTracker:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)
        
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)
        
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        
        self.first_detection = True

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
        prediction = self.kf.predict()
        prediction = prediction.flatten()
        return int(prediction[0]), int(prediction[1])

    def get_state(self):
        return int(self.kf.statePost[0, 0]), int(self.kf.statePost[1, 0])