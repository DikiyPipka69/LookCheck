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


# главный класс
class ClothingDetector:
    '''
    Класс для работы с моделью детекции одежды
    '''
    def __init__(self, model_path: str):
        # загружаем модель при создании объекта
        self.model = YOLO(model_path)

    def get_color(self, image: Image.Image, box) -> str:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w = x2 - x1
        h = y2 - y1

        # берём центральные 50%
        x1 = x1 + w // 4
        y1 = y1 + h // 4
        x2 = x2 - w // 4
        y2 = y2 - h // 4

        cropped = image.crop((x1, y1, x2, y2)).resize((50, 50))
        pixels = list(cropped.convert('RGB').getdata())

        # фильтруем телесные пиксели
        def is_skin(r, g, b):
            return (r > 95 and g > 40 and b > 20 and
                    max(r, g, b) - min(r, g, b) > 15 and
                    r > g and r > b and
                    abs(r - g) > 15)

        filtered = [p for p in pixels if not is_skin(p[0], p[1], p[2])]

        # если отфильтровали слишком много — используем все пиксели
        if len(filtered) < 20:
            filtered = pixels

        r = sum(p[0] for p in filtered) // len(filtered)
        g = sum(p[1] for p in filtered) // len(filtered)
        b = sum(p[2] for p in filtered) // len(filtered)

        rf, gf, bf = r/255, g/255, b/255
        cmax = max(rf, gf, bf)
        cmin = min(rf, gf, bf)
        diff = cmax - cmin
        l = (cmax + cmin) / 2
        s = 0 if diff == 0 else diff / (1 - abs(2*l - 1))

        if diff == 0:
            hue = 0
        elif cmax == rf:
            hue = 60 * (((gf - bf) / diff) % 6)
        elif cmax == gf:
            hue = 60 * (((bf - rf) / diff) + 2)
        else:
            hue = 60 * (((rf - gf) / diff) + 4)

        if l < 0.15:
            return "чёрный"
        if l > 0.85:
            return "белый"
        if s < 0.12:
            return "серый / белый"

        if hue < 15 or hue >= 345:
            return "красный / коричневый"
        if hue < 40:
            if l < 0.4:
                return "коричневый"
            return "оранжевый / бежевый"
        if hue < 70:
            return "жёлтый"
        if hue < 150:
            return "зелёный"
        if hue < 195:
            return "голубой"
        if hue < 250:
            return "синий"
        if hue < 290:
            return "фиолетовый"
        if hue < 345:
            return "розовый"

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

                detections.append({
                    "class": class_name,
                    "confidence": confidence,
                    "color": color
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
    def __init__(self):
        # список для хранения записей в памяти
        self.items = []

    def add(self, filename: str, image_url: str, detections: list):
        # добавляем новую запись в историю
        self.items.append({
            "id": uuid.uuid4().hex,
            "time": datetime.now().strftime("%H:%M %d.%m.%Y"),
            "filename": filename,
            "image_url": image_url,
            "detections": detections
        })

    def get_all(self) -> list:
        # возвращаем историю в обратном порядке (новые сверху)
        return list(reversed(self.items))

    def clear(self):
        # очищаем всю историю
        self.items = []

    def count(self) -> int:
        # возвращаем количество записей
        return len(self.items)


# инициализация приложения
app = FastAPI()

# создаём объекты наших классов
detector = ClothingDetector("runs/detect/train3/weights/best.pt")
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
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    image_b64 = base64.b64encode(contents).decode("utf-8")
    image_url = f"data:image/jpeg;base64,{image_b64}"
    image = Image.open(io.BytesIO(contents))

    detections, boxes_data = detector.detect(image)

    history_manager.add(file.filename, image_url, detections)

    return {"detections": detections, "boxes": boxes_data}


# эндпоинт для получения истории запросов
@app.get("/history")
async def get_history():
    return JSONResponse(content={"history": history_manager.get_all()})


# эндпоинт для очистки истории
@app.delete("/history")
async def clear_history():
    history_manager.clear()
    return {"message": "история очищена"}


# конструкция чтобы uvicorn работал пока не выключат
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)