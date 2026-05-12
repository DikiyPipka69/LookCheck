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
        # координаты рамки
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # берём только центральные 50% рамки чтобы избежать фона
        w = x2 - x1
        h = y2 - y1
        x1 = x1 + w // 4
        y1 = y1 + h // 4
        x2 = x2 - w // 4
        y2 = y2 - h // 4
        
        # вырезаем и уменьшаем
        cropped = image.crop((x1, y1, x2, y2)).resize((30, 30))
        
        # конвертируем в HSV для более точного определения цвета
        cropped_rgb = cropped.convert('RGB')
        pixels = list(cropped_rgb.getdata())
        
        # средний цвет
        r = sum(p[0] for p in pixels) // len(pixels)
        g = sum(p[1] for p in pixels) // len(pixels)
        b = sum(p[2] for p in pixels) // len(pixels)
        
        # нормализуем в 0-1
        rf, gf, bf = r/255, g/255, b/255
        
        cmax = max(rf, gf, bf)
        cmin = min(rf, gf, bf)
        diff = cmax - cmin
        
        # считаем яркость (lightness)
        l = (cmax + cmin) / 2
        
        # считаем насыщенность
        s = 0 if diff == 0 else diff / (1 - abs(2*l - 1))
        
        # считаем оттенок (hue)
        if diff == 0:
            h = 0
        elif cmax == rf:
            h = 60 * (((gf - bf) / diff) % 6)
        elif cmax == gf:
            h = 60 * (((bf - rf) / diff) + 2)
        else:
            h = 60 * (((rf - gf) / diff) + 4)
        
        # определяем цвет по HSL
        # сначала проверяем яркость и насыщенность
        if l < 0.15:
            return "чёрный"
        if l > 0.85:
            return "белый"
        if s < 0.12:
            if l < 0.35:
                return "тёмно-серый"
            if l > 0.65:
                return "светло-серый"
            return "серый"
        
        # определяем по оттенку
        if h < 15 or h >= 345:
            return "красный"
        if h < 40:
            return "оранжевый"
        if h < 70:
            return "жёлтый"
        if h < 150:
            return "зелёный"
        if h < 195:
            return "голубой"
        if h < 250:
            return "синий"
        if h < 290:
            return "фиолетовый"
        if h < 345:
            return "розовый"
        
        return "неизвестный"
    
    def detect(self, image: Image.Image) -> list:
        results = self.model(image, conf=0.5, iou=0.7)
        detections = []

        for r in results:
            if len(r.boxes) > 0:
                best_box = max(r.boxes, key=lambda b: float(b.conf[0]))
                color = self.get_color(image, best_box)
                detections.append({
                    "class": self.model.names[int(best_box.cls[0])],
                    "confidence": round(float(best_box.conf[0]) * 100, 1),
                    "color": color
                })

        return detections
    
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
    # читаем файл в память
    contents = await file.read()

    # конвертируем в base64 для превью в истории
    image_b64 = base64.b64encode(contents).decode("utf-8")
    image_url = f"data:image/jpeg;base64,{image_b64}"

    # открываем фото через pillow из памяти
    image = Image.open(io.BytesIO(contents))

    # прогоняем через детектор
    detections = detector.detect(image)

    # сохраняем в историю
    history_manager.add(file.filename, image_url, detections)

    # возвращаем результат на фронтенд
    return {"detections": detections}


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