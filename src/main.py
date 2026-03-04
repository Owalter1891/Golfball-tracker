import cv2
import argparse
import os
from detection import BallDetector
from tracking import KalmanTracker
from trajectory import TrajectoryAnalyzer
from visualization import Visualizer

def main(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    UPWARD_THRESHOLD_UPPER = -10
    MIN_UPWARD_FRAMES = 7
    candidate_tracks = []
    MAX_X_DIST = 5
    MAX_Y_DIST = 80
    locked_on = False
    last_frame = None

    detector = BallDetector()
    tracker = KalmanTracker()
    trajectory = TrajectoryAnalyzer()
    viz = Visualizer()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        last_frame = frame.copy()
        frame_count += 1
        
        motion_mask = detector.detect_motion_mask(frame)
        candidates = detector.detect(frame)
        
        filtered_candidates = []

        for (x, y, r) in candidates:
            if motion_mask[y, x] > 0:
                filtered_candidates.append((x, y, r))

        candidates = filtered_candidates

        if not locked_on:

            new_tracks = []

            for (x, y, r) in candidates:

                matched = False

                for track in candidate_tracks:

                    last_x, last_y, _ = track["positions"][-1]

                    dy = y - last_y
                    dx = abs(x - last_x)

                    if dx < MAX_X_DIST and abs(dy) < MAX_Y_DIST:

                        if dy < UPWARD_THRESHOLD_UPPER:
                            track["upward_frames"] += 1
                            track["positions"].append((x, y, frame_count))
                        
                        matched = True

                        if track["upward_frames"] >= MIN_UPWARD_FRAMES:
                            locked_on = True
                            tracker.update((x, y))
                            for x, y, pos_frame in track["positions"]:
                                tracker.update((x, y))
                                trajectory.add_point((x, y, pos_frame))
                            best_candidate = (x, y, r)
                            break

                if not matched:
                    new_tracks.append({
                        "positions": [(x, y, frame_count)],
                        "upward_frames": 0,
                    })

            candidate_tracks.extend(new_tracks)

            candidate_tracks = [
                t for t in candidate_tracks if len(t["positions"]) < 15
            ]
        elif locked_on:
            pred_x, pred_y = tracker.predict()
            min_dist = float('inf')

            for (x, y, r) in candidates:
                dy = abs(y - pred_y)
                dx = abs(x - pred_x)

                if dx < 10 and 5 < dy < 40:
                    tracker.update((x, y))
                    trajectory.add_point((x, y, frame_count))
                    viz.draw_ball(frame, x, y, r)

        #viz.draw_trajectory(frame, trajectory.points)
        viz.draw_predicted_trajectory(frame, trajectory, frame_count=frame_count)
            
        viz.draw_stats(frame, frame_count, fps, "")

        cv2.imshow('Golf Ball Tracker', frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Golf Ball Tracker")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output", type=str, default="results/output.mp4", help="Path to output video file")
    
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    main(args.video, args.output)