from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import shutil
import uuid
import os
from datetime import datetime

# инициализация приложения
app = FastAPI()

# модель
model = YOLO("runs/detect/train13/weights/best.pt")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# история запросов
history = []

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {})

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    temp_path = f"temp_{uuid.uuid4().hex}.jpg"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    results = model(temp_path)
    os.remove(temp_path)

    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "class": model.names[int(box.cls[0])],
                "confidence": round(float(box.conf[0]) * 100, 1)
            })

    # функция которая сохраняет историю
    history.append({
        "id": uuid.uuid4().hex,
        "time": datetime.now().strftime("%H:%M %d.%m.%Y"),
        "filename": file.filename,
        "detections": detections
    })

    return {"detections": detections}

# ручка для получения истории
@app.get("/history")
async def get_history():
    return JSONResponse(content={"history": list(reversed(history))})

# конструкция чтобы uvicorn работал пока не выключат
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=False)



