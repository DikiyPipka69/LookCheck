from ultralytics import YOLO

# Загружаем обученную модель
model = YOLO("runs/detect/train/weights/best.pt")

# Тестируем на фото
results = model("test.jpg")

# Показываем результат
for r in results:
    r.show()
    print(f"Найдено объектов: {len(r.boxes)}")
    for box in r.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])
        print(f"  {class_name}: {confidence:.2f}")














































































