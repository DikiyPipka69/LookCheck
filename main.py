from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import uuid
import base64
import io
from datetime import datetime
from PIL import Image
from collections import Counter
import time
# database
from database import SessionLocal, HistoryItem
from sqlalchemy.orm import Session
# security
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded



# главный класс
class ClothingDetector:
    '''
    Класс для классификации одежды
    '''
    def __init__(self, model_path: str, brand_model_path: str = None):
        self.model = YOLO(model_path)
        self.brand_model = YOLO(brand_model_path) if brand_model_path else None

    # приватный метод конвертации RGB в HSL
    def _rgb_to_hsl(self, r, g, b) -> tuple:
        rf, gf, bf = r/255, g/255, b/255
        cmax = max(rf, gf, bf)
        cmin = min(rf, gf, bf)
        diff = cmax - cmin
        l = (cmax + cmin) / 2
        s = 0 if diff == 0 else diff / (1 - abs(2*l - 1))
        if diff == 0:
            h = 0
        elif cmax == rf:
            h = 60 * (((gf - bf) / diff) % 6)
        elif cmax == gf:
            h = 60 * (((bf - rf) / diff) + 2)
        else:
            h = 60 * (((rf - gf) / diff) + 4)
        return h, s, l

    # функция распознавания цвета одежды
    def get_color(self, image: Image.Image, box) -> str:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w, h = x2 - x1, y2 - y1

        # берём центральные 50% рамки
        cropped = image.crop((
            x1 + w // 4, y1 + h // 4,
            x2 - w // 4, y2 - h // 4
        )).resize((50, 50))

        pixels = list(cropped.convert('RGB').getdata())

        # фильтруем телесные пиксели
        def is_skin(r, g, b):
            return (r > 95 and g > 40 and b > 20 and
                    max(r, g, b) - min(r, g, b) > 15 and
                    r > g and r > b and abs(r - g) > 15)

        filtered = [p for p in pixels if not is_skin(p[0], p[1], p[2])]
        if len(filtered) < 20:
            filtered = pixels

        # средний цвет
        r = sum(p[0] for p in filtered) // len(filtered)
        g = sum(p[1] for p in filtered) // len(filtered)
        b = sum(p[2] for p in filtered) // len(filtered)

        hue, s, l = self._rgb_to_hsl(r, g, b)

        # определяем цвет по яркости и оттенку
        if l < 0.15: return "чёрный"
        if l > 0.85: return "белый"
        if s < 0.12: return "серый"
        if hue < 15 or hue >= 345: return "красный"
        if hue < 40: return "коричневый" if l < 0.4 else "оранжевый"
        if hue < 70: return "жёлтый"
        if hue < 150: return "зелёный"
        if hue < 195: return "голубой"
        if hue < 250: return "синий"
        if hue < 290: return "фиолетовый"
        if hue < 345: return "розовый"
        return "неизвестный"
    
    def detect(self, image: Image.Image) -> tuple:
        results = self.model(image)  # conf=0.5, iou=0.7
        detections = []
        boxes_data = []

        for r in results:
            img_w, img_h = image.size
            for box in r.boxes:
                color = self.get_color(image, box)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_name = self.model.names[int(box.cls[0])]
                confidence = round(float(box.conf[0]) * 100, 1)

                # определяем бренд если модель загружена
                brand = None
                if self.brand_model is not None:
                    # вырезаем область одежды и ищем бренд внутри
                    cropped = image.crop((x1, y1, x2, y2))
                    brand_results = self.brand_model(cropped)
                    for br in brand_results:
                        if len(br.boxes) > 0:
                            best_brand = max(br.boxes, key=lambda b: float(b.conf[0]))
                            if float(best_brand.conf[0]) > 0.5:
                                brand = self.brand_model.names[int(best_brand.cls[0])]

                detections.append({
                    "class": class_name,
                    "confidence": confidence,
                    "color": color,
                    "brand": brand
                })

                boxes_data.append({
                    "x1": x1 / img_w,
                    "y1": y1 / img_h,
                    "x2": x2 / img_w,
                    "y2": y2 / img_h,
                    "label": f"{class_name} ({color}) {confidence}%"
                })

        return detections, boxes_data
    
    def get_class_names(self) -> list:
        # возвращаем список всех классов модели
        return list(self.model.names.values())


# класс для управления историей запросов
class HistoryManager:
    def add(self, filename: str, image_url: str, detections: list, process_time: float):
        # сохраняем запись в базу данных
        db: Session = SessionLocal()
        try:
            item = HistoryItem(
                filename=filename,
                image_url=image_url,
                detections=detections,
                process_time=round(process_time, 2)
            )
            db.add(item)
            db.commit()
        finally:
            db.close()

    def get_all(self) -> list:
        # получаем всю историю из базы
        db: Session = SessionLocal()
        try:
            items = db.query(HistoryItem).order_by(HistoryItem.time.desc()).all()
            return [
                {
                    "id": item.id,
                    "time": item.time.strftime("%H:%M %d.%m.%Y"),
                    "filename": item.filename,
                    "image_url": item.image_url,
                    "detections": item.detections,
                    "process_time": item.process_time
                }
                for item in items
            ]
        finally:
            db.close()

    def get_stats(self) -> dict:
        # считаем статистику из базы
        db: Session = SessionLocal()
        try:
            items = db.query(HistoryItem).all()

            if not items:
                return {
                    "total": 0,
                    "class_counts": {},
                    "avg_confidence": 0,
                    "avg_process_time": 0,
                    "color_counts": {}
                }

            class_counts = Counter()
            color_counts = Counter()
            confidences = []
            process_times = []

            for item in items:
                process_times.append(item.process_time or 0)
                for d in item.detections:
                    class_counts[d["class"]] += 1
                    color_counts[d.get("color", "неизвестный")] += 1
                    confidences.append(d["confidence"])

            return {
                "total": len(items),
                "class_counts": dict(class_counts.most_common()),
                "color_counts": dict(color_counts.most_common()),
                "avg_confidence": round(sum(confidences) / len(confidences), 1) if confidences else 0,
                "avg_process_time": round(sum(process_times) / len(process_times), 2) if process_times else 0,
                "total_detections": len(confidences)
            }
        finally:
            db.close()

    def clear(self):
        # очищаем всю историю
        db: Session = SessionLocal()
        try:
            db.query(HistoryItem).delete()
            db.commit()
        finally:
            db.close()

    def count(self) -> int:
        db: Session = SessionLocal()
        try:
            return db.query(HistoryItem).count()
        finally:
            db.close()


# инициализация приложения
app = FastAPI()

# rate limiting — не более 10 запросов в минуту с одного IP
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# создаём объекты наших классов
detector = ClothingDetector(
    "runs/detect/train3/weights/best.pt",
    "runs/detect/runs_brand/brand_detector/weights/best.pt"
)
history_manager = HistoryManager()

# подключаем папку со статикой (css, js)
app.mount("/static", StaticFiles(directory="static"), name="static")

# подключаем шаблоны html
templates = Jinja2Templates(directory="templates")


# главная страница
@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {})


# эндпоинт для определения одежды на фото
@app.post("/detect")
@limiter.limit("10/minute")
async def detect(request: Request, file: UploadFile = File(...)):
    start_time = time.time()

    contents = await file.read()
    image_b64 = base64.b64encode(contents).decode("utf-8")
    image_url = f"data:image/jpeg;base64,{image_b64}"
    image = Image.open(io.BytesIO(contents))

    detections, boxes_data = detector.detect(image)

    process_time = time.time() - start_time
    history_manager.add(file.filename, image_url, detections, process_time)

    return {"detections": detections, "boxes": boxes_data}


# эндпоинт для получения истории запросов
@app.get("/history")
async def get_history():
    return JSONResponse(content={"history": history_manager.get_all()})


# эндпоинт для получения статистики
@app.get("/stats")
async def get_stats():
    return JSONResponse(content={"stats": history_manager.get_stats()})


# эндпоинт для очистки истории
@app.delete("/history")
async def clear_history():
    history_manager.clear()
    return {"message": "история очищена"}


# конструкция чтобы uvicorn работал пока не выключат
if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0" if os.environ.get("RAILWAY_ENVIRONMENT") else "127.0.0.1"
    uvicorn.run("main:app", host=host, port=port, reload=False)