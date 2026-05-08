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

model = YOLO("runs/detect/train/weights/best.pt")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

history = []

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {})

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # читаем файл в память
    contents = await file.read()
    
    # фигачим в base64 для превью
    image_b64 = base64.b64encode(contents).decode("utf-8")
    image_url = f"data:image/jpeg;base64,{image_b64}"

    # прогоняем через модель из памяти
    image = Image.open(io.BytesIO(contents))
    results = model(image)

    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "class": model.names[int(box.cls[0])],
                "confidence": round(float(box.conf[0]) * 100, 1)
            })

    history.append({
        "id": uuid.uuid4().hex,
        "time": datetime.now().strftime("%H:%M %d.%m.%Y"),
        "filename": file.filename,
        "image_url": image_url,
        "detections": detections
    })

    return {"detections": detections}

@app.get("/history")
async def get_history():
    return JSONResponse(content={"history": list(reversed(history))})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)