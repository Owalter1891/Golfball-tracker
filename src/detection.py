import cv2
from ultralytics import YOLO


class BallDetector:
    def __init__(self, ball_model_path="best.pt", club_model_path="best.pt"):

        self.ball_model = YOLO(ball_model_path)
        self.club_model = YOLO(club_model_path)

        self.club_head_class_id = 3

    def detect_ball(self, frame):
        """
        Detect all golf balls with confidence > 0.5.
        Returns a list of tuples: [(x, y, r, conf), ...]
        """
        frame_h, frame_w = frame.shape[:2]
        frame_area = frame_h * frame_w

        results = self.ball_model(frame, verbose=False)
        detections = []

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                conf = float(box.conf[0])

                if conf < 0.25:
                    continue

                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                box_w = x2 - x1
                box_h = y2 - y1

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                r = int(max(box_w, box_h) / 2)

                detections.append((cx, cy, r, conf))

        return detections

    def detect_club_head(self, frame):
        """
        Detect the highest-confidence club head.
        Returns (x, y, r, conf) or None
        """
        results = self.club_model(frame, verbose=False)

        best_head = None
        best_conf = 0.0

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if cls != self.club_head_class_id:
                    continue

                if conf <= best_conf:
                    continue

                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                r = int(max(x2 - x1, y2 - y1) / 2)

                best_head = (cx, cy, r, conf)
                best_conf = conf

        return best_head