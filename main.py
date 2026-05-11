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

app = FastAPI()

# обученная моделька YOLO
model = YOLO("runs/detect/train3/weights/best.pt")

# подключаем папку со статикой (css, js)
app.mount("/static", StaticFiles(directory="static"), name="static")

# подключаем шаблоны html
templates = Jinja2Templates(directory="templates")

# хранилище истории запросов в памяти
history = []

# главная страница
@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {})

# эндпоинт для определения одежды на фото
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # читаем файл в память
    contents = await file.read()
    
    # фигачим в base64 для превью
    image_b64 = base64.b64encode(contents).decode("utf-8")
    image_url = f"data:image/jpeg;base64,{image_b64}"

    # прогоняем через модель из памяти
    image = Image.open(io.BytesIO(contents))
    results = model(image, conf=0.2, iou=0.7)

    # собираем результаты детекции
    detections = []
    for r in results:
        if len(r.boxes) > 0:
            best_box = max(r.boxes, key=lambda b: float(b.conf[0]))
            detections.append({
                "class": model.names[int(best_box.cls[0])],
                "confidence": round(float(best_box.conf[0]) * 100, 1)
            })

    # сохраняем запрос в историю
    history.append({
        "id": uuid.uuid4().hex,
        "time": datetime.now().strftime("%H:%M %d.%m.%Y"),
        "filename": file.filename,
        "image_url": image_url,
        "detections": detections
    })

    # возвращаем результат на фронтенд
    return {"detections": detections}

# получаем истории
@app.get("/history")
async def get_history():
    return JSONResponse(content={"history": list(reversed(history))})

# конструкция чтобы uvicorn работал пока не выключат
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)