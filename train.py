from ultralytics import YOLO

model = YOLO("yolo26n.pt")

model.train(
    data="dataset/data.yaml",
    epochs=20,
    imgsz=640,
    patience=5
)














































































