import cv2
import numpy as np

class Visualizer:
    def draw_ball(self, frame, x, y, r, color=(0, 255, 0)):
        cv2.circle(frame, (int(x), int(y)), int(r), color, 2)
        cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), 3)

    def draw_predicted_trajectory(self, frame, trajectory, steps=150, history_color=(0, 255, 0), pred_color=(0, 0, 255)):
        if len(trajectory.points) > 1:
            historical_points = np.array([[p[0], p[1]] for p in trajectory.points], dtype=np.int32)
            cv2.polylines(frame, [historical_points], isClosed=False, color=history_color, thickness=2)

        predicted_points = trajectory.predict_next_points(steps)
        if len(predicted_points) > 1:
            future_points = np.array([[p[0], p[1]] for p in predicted_points], dtype=np.int32)
            cv2.polylines(frame, [future_points], isClosed=False, color=pred_color, thickness=2)

    def draw_stats(self, frame, frame_count, fps, state_msg):
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)