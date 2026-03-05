from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8s.pt")
    model.train(
        data=r"C:\Users\oscar\Golfball-tracker\data\golfBall.v1i.yolov8\data.yaml",
        epochs=50,
        imgsz=960,
        batch=16,
        device="cuda",
        workers=4
    )