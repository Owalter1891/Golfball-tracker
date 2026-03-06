import cv2
import argparse
import os
import numpy as np
from detection import BallDetector
from tracking import KalmanTracker
from trajectory import Trajectory
from visualization import Visualizer


def main(video_path, output_path, track_club_head=False):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector = BallDetector(ball_model_path="../runs/detect/train2/weights/best.pt", club_model_path="../runs/detect/train/weights/best.pt")
    tracker = KalmanTracker()
    ball_path = Trajectory()
    viz = Visualizer()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    last_raw_pos = None
    moving = False
    frames_tracked = 0

    MOVE_THRESHOLD = 4
    MAX_JUMP_DISTANCE = 75
    FRAMES_TRACKED_THRESHOLD = 20

    club_head_points = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        ball_detections = detector.detect_ball(frame)

        if moving:
            pred_x, pred_y = tracker.predict()

            if frames_tracked < FRAMES_TRACKED_THRESHOLD:
                matched_ball = None
                min_dist = float('inf')

                for det in ball_detections:
                    x_det, y_det, _, _ = det
                    dist = np.sqrt((x_det - pred_x) ** 2 + (y_det - pred_y) ** 2)

                    if dist < MAX_JUMP_DISTANCE and dist < min_dist:
                        min_dist = dist
                        matched_ball = det

                if matched_ball is not None:
                    x, y, r, _ = matched_ball
                    tracker.update((x, y))
                    frames_tracked += 1
                    
                    kx, ky = tracker.get_state()
                    ball_path.add_point((kx, ky))
                    viz.draw_ball(frame, kx, ky, r)
                else:
                    ball_path.add_point((pred_x, pred_y))
            
            else:
                ball_path.add_point((pred_x, pred_y))
        
        else:
            best_ball = None
            if ball_detections:
                ball_detections.sort(key=lambda x: x[3], reverse=True)
                best_ball = ball_detections[0]

            if best_ball is not None:
                x, y, r, _ = best_ball
                
                if last_raw_pos is not None:
                    dist = np.sqrt((x - last_raw_pos[0])**2 + (y - last_raw_pos[1])**2)
                    dy = last_raw_pos[1] - y 

                    if dist > MOVE_THRESHOLD and dy > 5:
                        moving = True
                        tracker.update((x, y))
                        frames_tracked = 1
                        kx, ky = tracker.get_state()
                        ball_path.add_point((kx, ky))
                
                last_raw_pos = (x, y)
                viz.draw_ball(frame, x, y, r)

        if track_club_head:
            club_head = detector.detect_club_head(frame)
            if club_head is not None:
                cx, cy, r, conf = club_head
                club_head_points.append((int(cx), int(cy)))

            if len(club_head_points) > 1:
                points_array = np.array(club_head_points, dtype=np.int32)
                cv2.polylines(frame, [points_array], isClosed=False, color=(0, 255, 255), thickness=2)

        if moving:
            viz.draw_trajectory(frame, ball_path.points)

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