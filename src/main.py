import cv2
import argparse
import os
import numpy as np
from detection import BallDetector
from tracking import KalmanTracker
from trajectory import TrajectoryAnalyzer
from visualization import Visualizer


def main(video_path, output_path, track_club_head=False):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector = BallDetector("../runs/detect/train/weights/best.pt")
    tracker = KalmanTracker()
    trajectory = TrajectoryAnalyzer()
    viz = Visualizer()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    last_position = None
    moving = False

    MOVE_THRESHOLD = 4
    MAX_JUMP_DISTANCE = 75

    club_head_points = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        ball_detection = detector.detect_ball(frame)

        if ball_detection is not None and moving:
            pred_x, pred_y = tracker.predict()
            x, y, _, _ = ball_detection
            dist = np.sqrt((x - pred_x) ** 2 + (y - pred_y) ** 2)
            if dist > MAX_JUMP_DISTANCE:
                ball_detection = None

        if ball_detection is not None:
            x, y, r, conf = ball_detection

            if last_position is not None and not moving:
                dist = np.sqrt((x - last_position[0]) ** 2 + (y - last_position[1]) ** 2)
                if dist > MOVE_THRESHOLD:
                    moving = True

            last_position = (x, y)
            tracker.update((x, y))

            if moving:
                kx, ky = tracker.get_state()
                trajectory.add_point((kx, ky, frame_count))
                viz.draw_ball(frame, kx, ky, r)
            else:
                viz.draw_ball(frame, x, y, r)
        elif moving:
            pred_x, pred_y = tracker.predict()
            trajectory.add_point((pred_x, pred_y, frame_count))

        if track_club_head:
            club_head = detector.detect_club_head(frame)
            if club_head is not None:
                cx, cy, r, conf = club_head
                club_head_points.append((int(cx), int(cy)))

            if len(club_head_points) > 1:
                points_array = np.array(club_head_points, dtype=np.int32)
                cv2.polylines(frame, [points_array], isClosed=False, color=(0, 255, 255), thickness=2)

        viz.draw_predicted_trajectory(frame, trajectory, steps=100)
        viz.draw_stats(frame, frame_count, fps, "")

        cv2.imshow("Golf Ball Tracker", frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Golf Ball Tracker")

    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output", type=str, default="results/output.mp4", help="Path to output video file")
    parser.add_argument("--track_club_head", action="store_true", help="Draw a trail for the detected club head")

    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    main(args.video, args.output, track_club_head=args.track_club_head)