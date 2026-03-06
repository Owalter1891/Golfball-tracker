import cv2
import numpy as np
from collections import deque


class Visualizer:
    def draw_ball(self, frame, x, y, r, color=(0, 255, 0)):
        cv2.circle(frame, (int(x), int(y)), max(int(r), 3), color, 2)
        cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), 3)

    def draw_trajectory(self, frame, historical_points, history_color=(0, 120, 255)):
        """
        Draws a fading trajectory trail.

        Segments near the tail (oldest) are thin and dim; segments near the
        head (newest) are thick and bright. This gives a natural motion-blur
        look without any alpha compositing.
        """
        pts = list(historical_points)
        n = len(pts)
        if n < 2:
            return

        for i in range(1, n):
            t = i / (n - 1)

            r_c = int(history_color[2] * t)
            g_c = int(history_color[1] * t)
            b_c = int(history_color[0] * t)
            color = (b_c, g_c, r_c)

            thickness = max(1, int(1 + 3 * t))

            cv2.line(
                frame,
                (int(pts[i - 1][0]), int(pts[i - 1][1])),
                (int(pts[i][0]), int(pts[i][1])),
                color,
                thickness,
                lineType=cv2.LINE_AA,
            )

    def draw_stats(self, frame, frame_count, fps, state_msg):
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if state_msg:
            cv2.putText(frame, state_msg, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)