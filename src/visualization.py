import cv2
import numpy as np

class Visualizer:
    def draw_ball(self, frame, x, y, r, color=(0, 255, 0)):
        cv2.circle(frame, (x, y), r, color, 2)
        cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)

    def draw_trajectory(self, frame, points, color=(0, 255, 255)):
        if len(points) < 2:
            return
        for i in range(1, len(points)):
            cv2.line(frame, points[i - 1], points[i], color, 2)

    def draw_predicted_trajectory(self, frame, trajectory, steps=350, color=(0,0,255), frame_count = 0):
        predicted_points = trajectory.predict_next_points(steps)
        if len(predicted_points) < 2:
            return
        total_points = trajectory.points + predicted_points
        for i in range(1, len(total_points)):
            x, y, point_frame = total_points[i]
            print("Point_Frame: ", point_frame)

            print("frame_Count: ", frame_count)
            x2, y2, _ = total_points[i-1]
            if frame_count < point_frame:
                break
            cv2.line(frame, (x2, y2), (x, y), color, 2)
            

    def draw_stats(self, frame, frame_count, fps, state_msg):
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)