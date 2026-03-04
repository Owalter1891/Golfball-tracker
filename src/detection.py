import cv2
import numpy as np

class BallDetector:
    def __init__(self):
        self.dp = 1.2
        self.min_dist = 50
        self.param1 = 210
        self.param2 = 15
        self.min_radius = 1
        self.max_radius = 15
        
        self.back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

    def preprocess(self, frame):
        """
        Convert to grayscale and apply Gaussian blur.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        return blurred

    def detect(self, frame):
        """
        Returns a list of (x, y, r) tuples.
        """
        processed_frame = self.preprocess(frame)
        
        circles = cv2.HoughCircles(
            processed_frame, 
            cv2.HOUGH_GRADIENT, 
            dp=self.dp, 
            minDist=self.min_dist,
            param1=self.param1, 
            param2=self.param2, 
            minRadius=self.min_radius, 
            maxRadius=self.max_radius
        )

        detected_balls = []
        if circles is not None:
            circles = np.around(circles).astype(int)
            for i in circles[0, :]:
                detected_balls.append((i[0], i[1], i[2]))
        
        return detected_balls

    def detect_motion_mask(self, frame):
        """
        Returns a binary mask of moving objects.
        """
        fg_mask = self.back_sub.apply(frame)
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        return thresh

if __name__ == "__main__":
    import argparse

    def nothing(x):
        pass

    parser = argparse.ArgumentParser(description="Tune Ball Detection Parameters")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error opening video: {args.video}")
        exit()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    ret, current_frame = cap.read()
    if not ret:
        print("Error reading first frame")
        exit()

    detector = BallDetector()

    cv2.namedWindow("Tuning")
    
    cv2.createTrackbar("Frame", "Tuning", 0, total_frames - 1, nothing)
    cv2.createTrackbar("dp x10", "Tuning", int(detector.dp * 10), 50, nothing)
    cv2.createTrackbar("minDist", "Tuning", detector.min_dist, 200, nothing)
    cv2.createTrackbar("param1", "Tuning", detector.param1, 300, nothing)
    cv2.createTrackbar("param2", "Tuning", detector.param2, 100, nothing)
    cv2.createTrackbar("minRadius", "Tuning", detector.min_radius, 100, nothing)
    cv2.createTrackbar("maxRadius", "Tuning", detector.max_radius, 100, nothing)

    current_frame_idx = 0

    while True:
        trackbar_frame_idx = cv2.getTrackbarPos("Frame", "Tuning")
        if trackbar_frame_idx != current_frame_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, trackbar_frame_idx)
            ret, frame = cap.read()
            if ret:
                current_frame = frame
                current_frame_idx = trackbar_frame_idx
        dp_val = cv2.getTrackbarPos("dp x10", "Tuning")
        detector.dp = max(0.1, dp_val / 10.0)
        detector.min_dist = max(1, cv2.getTrackbarPos("minDist", "Tuning"))
        detector.param1 = max(1, cv2.getTrackbarPos("param1", "Tuning"))
        detector.param2 = max(1, cv2.getTrackbarPos("param2", "Tuning"))
        detector.min_radius = cv2.getTrackbarPos("minRadius", "Tuning")
        detector.max_radius = cv2.getTrackbarPos("maxRadius", "Tuning")

        candidates = detector.detect(current_frame)

        display = current_frame.copy()
        
        for (x, y, r) in candidates:
            cv2.circle(display, (x, y), r, (0, 255, 0), 2)
            cv2.circle(display, (x, y), 2, (0, 0, 255), 3)

        info_text = [
            f"dp: {detector.dp:.1f}",
            f"minDist: {detector.min_dist}",
            f"param1: {detector.param1}",
            f"param2: {detector.param2}",
            f"minR: {detector.min_radius}",
            f"maxR: {detector.max_radius}",
            f"Detected: {len(candidates)}"
        ]
        
        for i, line in enumerate(info_text):
            cv2.putText(display, line, (10, 30 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Tuning", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()