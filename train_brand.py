from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO("yolo26n.pt")
    model.train(
        data="dataset_brand/data.yaml",
        epochs=50,
        imgsz=640,
        patience=5,
        batch=16,
        device="cuda",
        project="runs_brand",
        name="brand_detector"
    )