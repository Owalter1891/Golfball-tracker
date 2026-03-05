import cv2
from ultralytics import YOLO


class BallDetector:
    def __init__(self, model_path="best.pt"):

        self.model = YOLO(model_path)

        # class index for golf ball
        self.ball_class_id = 1
        self.club_head_class_id = 3

    def detect_ball(self, frame):
        """
        Detect the highest-confidence golf ball.
        Returns (x, y, r, conf) or None
        """

        results = self.model(frame, verbose=False)

        best_ball = None
        best_conf = 0

        for r in results:

            if r.boxes is None:
                continue

            for box in r.boxes:

                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if cls != self.ball_class_id:
                    continue

                if conf < best_conf:
                    continue

                x1, y1, x2, y2 = box.xyxy[0]

                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                r = int(max(x2 - x1, y2 - y1) / 2)

                best_ball = (cx, cy, r, conf)
                best_conf = conf

        return best_ball
    
    def detect_club_head(self, frame):
        """
        Detect the highest-confidence club head.
        Returns (x, y, r, conf) or None
        """

        results = self.model(frame, verbose=False)

        best_head = None
        best_conf = 0

        for r in results:

            if r.boxes is None:
                continue

            for box in r.boxes:

                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if cls != self.club_head_class_id:
                    continue

                if conf < best_conf:
                    continue

                x1, y1, x2, y2 = box.xyxy[0]

                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                r = int(max(x2 - x1, y2 - y1) / 2)

                best_head = (cx, cy, r, conf)
                best_conf = conf

        return best_head