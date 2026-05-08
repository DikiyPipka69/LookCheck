from ultralytics import YOLO
# модель
model = YOLO("yolo26n.pt")
# обучение
model.train(
    data="dataset/data.yaml",
    epochs=20,
    imgsz=640,
    patience=5
)