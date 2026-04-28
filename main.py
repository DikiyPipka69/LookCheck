from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from ultralytics import YOLO
import shutil
import uuid
import os

app = FastAPI()

model = YOLO("runs/detect/train13/weights/best.pt")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {})

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # Сохраняем фото временно
    temp_path = f"temp_{uuid.uuid4().hex}.jpg"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Прогоняем через модель
    results = model(temp_path)
    os.remove(temp_path)

    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "class": model.names[int(box.cls[0])],
                "confidence": round(float(box.conf[0]) * 100, 1)
            })

    return {"detections": detections}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)


























































