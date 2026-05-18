import os
import sys
import time
import uuid
import io
import base64
from datetime import datetime
from collections import Counter
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from ultralytics import YOLO
# Импорты для работы с базой данных
from database import SessionLocal, HistoryItem
from sqlalchemy.orm import Session
# Импорты для защиты от спама и ограничения частоты запросов (Rate Limiting)
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded


# =====================================================================
# КЛАСС ДЛЯ РАБОТЫ С НЕЙРОСЕТЯМИ (YOLO)
# =====================================================================
class ClothingDetector:
    """
    Класс отвечает за загрузку моделей компьютерного зрения YOLO,
    распознавание типов одежды, определение их цвета и поиск брендов.
    """
    def __init__(self, model_path: str, brand_model_path: str = None):
        """
        Инициализация класса и загрузка весов нейросетей в память.
        :param model_path: Путь к основной модели распознавания одежды.
        :param brand_model_path: Путь к опциональной модели распознавания брендов.
        """
        # Загружаем основную модель YOLO для поиска типов одежды
        self.model = YOLO(model_path)
        
        # Если передан путь к брендовой модели, загружаем её, иначе оставляем None
        self.brand_model = YOLO(brand_model_path) if brand_model_path else None

    def _rgb_to_hsl(self, r: int, g: int, b: int) -> tuple:
        """
        Вспомогательный приватный метод для перевода цвета из формата RGB в HSL.
        HSL (Hue, Saturation, Lightness) позволяет точнее определять оттенки.
        """
        # Переводим значения пикселей из диапазона 0-255 в диапазон 0.0-1.0
        rf, gf, bf = r / 255.0, g / 255.0, b / 255.0
        cmax = max(rf, gf, bf)  # Максимальное значение среди R, G, B
        cmin = min(rf, gf, bf)  # Минимальное значение среди R, G, B
        diff = cmax - cmin      # Разница между макс. и мин. (контрастность)
        
        # Расчет яркости (Lightness)
        l = (cmax + cmin) / 2
        
        # Расчет насыщенности (Saturation)
        s = 0 if diff == 0 else diff / (1 - abs(2 * l - 1))
        
        # Расчет оттенка в градусах от 0 до 360 (Hue)
        if diff == 0:
            h = 0
        elif cmax == rf:
            h = 60 * (((gf - bf) / diff) % 6)
        elif cmax == gf:
            h = 60 * (((bf - rf) / diff) + 2)
        else:
            h = 60 * (((rf - gf) / diff) + 4)
            
        return h, s, l

    def get_color(self, image: Image.Image, box) -> str:
        """
        Определяет преобладающий цвет одежды внутри рамки (bounding box).
        """
        # Получаем координаты рамки объекта (переводим тензоры в обычные int)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w, h = x2 - x1, y2 - y1

        # Вырезаем центральную область (50% от ширины и высоты), 
        # чтобы уменьшить влияние фона вокруг одежды, и сжимаем до 50x50 для скорости
        cropped = image.crop((
            x1 + w // 4, y1 + h // 4,
            x2 - w // 4, y2 - h // 4
        )).resize((50, 50))

        # Получаем список всех RGB пикселей из вырезанного кусочка
        pixels = list(cropped.convert('RGB').getdata())

        def is_skin(r, g, b):
            """Внутренняя функция-фильтр для отсечения оттенков человеческой кожи"""
            return (r > 95 and g > 40 and b > 20 and
                    max(r, g, b) - min(r, g, b) > 15 and
                    r > g and r > b and abs(r - g) > 15)

        # Отфильтровываем пиксели: оставляем только те, которые НЕ похожи на кожу
        filtered = [p for p in pixels if not is_skin(p[0], p[1], p[2])]
        
        # Предохранитель: если после фильтрации почти ничего не осталось, берем исходные пиксели
        if len(filtered) < 20:
            filtered = pixels

        # Находим среднее арифметическое значение цвета по всем оставшимся пикселям
        r = sum(p[0] for p in filtered) // len(filtered)
        g = sum(p[1] for p in filtered) // len(filtered)
        b = sum(p[2] for p in filtered) // len(filtered)

        # Переводим средний RGB в систему HSL для удобной классификации текстом
        hue, s, l = self._rgb_to_hsl(r, g, b)

        # Правила определения цвета по яркости (l), насыщенности (s) и оттенку (hue)
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
        """
        Запускает поиск одежды на изображении, определяет цвет и ищет бренды.
        """
        results = self.model(image)  # Прогон картинки через основную модель YOLO
        detections = []              # Сюда пишем текстовые данные для статистики/базы
        boxes_data = []              # Сюда пишем координаты для отрисовки на фронтенде

        for r in results:
            img_w, img_h = image.size # Оригинальные размеры изображения
            for box in r.boxes:
                # 1. Определяем цвет текущего найденного объекта одежды
                color = self.get_color(image, box)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # 2. Получаем текстовое имя класса одежды и уверенность модели
                class_name = self.model.names[int(box.cls[0])]
                confidence = round(float(box.conf[0]) * 100, 1)

                # 3. Каскадное определение бренда (если подключена вторая модель)
                brand = None
                if self.brand_model is not None:
                    # Вырезаем объект одежды целиком
                    cropped = image.crop((x1, y1, x2, y2))
                    brand_results = self.brand_model(cropped)
                    
                    for br in brand_results:
                        if len(br.boxes) > 0:
                            # Ищем логотип бренда внутри одежды с наивысшим скором (уверенностью)
                            best_brand = max(br.boxes, key=lambda b: float(b.conf[0]))
                            # Если уверенность распознавания бренда выше 50%, сохраняем его имя
                            if float(best_brand.conf[0]) > 0.5:
                                brand = self.brand_model.names[int(best_brand.cls[0])]

                # Формируем структуру данных для истории в БД
                detections.append({
                    "class": class_name,
                    "confidence": confidence,
                    "color": color,
                    "brand": brand
                })

                # Формируем структуру с ОТНОСИТЕЛЬНЫМИ координатами (от 0 до 1) для фронтенда
                boxes_data.append({
                    "x1": x1 / img_w,
                    "y1": y1 / img_h,
                    "x2": x2 / img_w,
                    "y2": y2 / img_h,
                    "label": f"{class_name} ({color}) {confidence}%" + (f" [{brand}]" if brand else "")
                })

        return detections, boxes_data
    
    def get_class_names(self) -> list:
        """Возвращает список всех категорий одежды, которые знает модель"""
        return list(self.model.names.values())


# =====================================================================
# КЛАСС ДЛЯ РАБОТЫ С ИСТОРИЕЙ И СТАТИСТИКОЙ (БАЗА ДАННЫХ)
# =====================================================================
class HistoryManager:
    """
    Класс инкапсулирует в себе все операции с SQLAlchemy ORM:
    добавление записей, чтение логов, сбор агрегированной статистики и очистку.
    """
    def add(self, filename: str, image_url: str, detections: list, process_time: float):
        """Сохраняет результат нового сканирования в базу данных"""
        db: Session = SessionLocal()  # Открываем сессию подключения к БД
        try:
            item = HistoryItem(
                filename=filename,
                image_url=image_url,
                detections=detections,
                process_time=round(process_time, 2)
            )
            db.add(item)       # Кладим объект в очередь на добавление
            db.commit()        # Сохраняем транзакцию в файл БД
        except Exception as e:
            db.rollback()      # Если произошла ошибка (например, сбой диска) — откатываемся
            raise e
        finally:
            db.close()         # Обязательно закрываем соединение, чтобы избежать утечки пула

    def get_all(self) -> list:
        """Возвращает список всех прошлых запросов, отсортированных от новых к старым"""
        db: Session = SessionLocal()
        try:
            items = db.query(HistoryItem).order_by(HistoryItem.time.desc()).all()
            return [
                {
                    "id": item.id,
                    "time": item.time.strftime("%H:%M %d.%m.%Y"),
                    "filename": item.filename,
                    "image_url": item.image_url,  # Картинка в формате base64 string
                    "detections": item.detections,
                    "process_time": item.process_time
                }
                for item in items
            ]
        finally:
            db.close()

    # СТАТИСТИКА
    def get_stats(self) -> dict:
        """Вычисляет аналитическую статистику по всей базе данных для дашборда"""
        db: Session = SessionLocal()
        try:
            items = db.query(HistoryItem).all()

            # Если база пустая, отдаем дефолтные нулевые значения
            if not items:
                return {
                    "total": 0,
                    "class_counts": {},
                    "avg_confidence": 0,
                    "avg_process_time": 0,
                    "color_counts": {},
                    "total_detections": 0
                }

            # Инициализируем счетчики
            class_counts = Counter()
            color_counts = Counter()
            confidences = []
            process_times = []

            # Проходим по всем строкам базы данных и собираем массивы данных
            for item in items:
                process_times.append(item.process_time or 0)
                for d in item.detections:
                    class_counts[d["class"]] += 1
                    color_counts[d.get("color", "неизвестный")] += 1
                    confidences.append(d["confidence"])

            # Формируем итоговый аналитический отчет
            return {
                "total": len(items),                                     # Сколько картинок загружено всего
                "class_counts": dict(class_counts.most_common()),        # Топ категорий одежды (от частых к редким)
                "color_counts": dict(color_counts.most_common()),        # Топ цветов одежды
                "avg_confidence": round(sum(confidences) / len(confidences), 1) if confidences else 0, # Средняя уверенность ИИ
                "avg_process_time": round(sum(process_times) / len(process_times), 2) if process_times else 0, # Средняя скорость работы
                "total_detections": len(confidences)                     # Общее количество распознанных вещей
            }
        finally:
            db.close()

    def clear(self):
        """Полностью удаляет все строки из таблицы истории"""
        db: Session = SessionLocal()
        try:
            db.query(HistoryItem).delete()
            db.commit()
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()

    def count(self) -> int:
        """Возвращает количество всех записей в базе данных"""
        db: Session = SessionLocal()
        try:
            return db.query(HistoryItem).count()
        finally:
            db.close()


# =====================================================================
# ИНИЦИАЛИЗАЦИЯ И НАСТРОЙКА WEB-ПРИЛОЖЕНИЯ FASTAPI
# =====================================================================
app = FastAPI(title="Clothing Detection API")

# Настройка лимитера частоты запросов (Rate Limiting)
# Запоминает клиентов по их IP-адресу (get_remote_address)
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
# Регистрируем глобальный обработчик ошибок превышения лимита (вернет HTTP 429)
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Создаем экземпляры управляющих классов и загружаем веса YOLO нейросетей
detector = ClothingDetector(
    "runs/detect/train3/weights/best.pt",
    "runs/detect/runs_brand/brand_detector/weights/best.pt"
)
history_manager = HistoryManager()

# Монтируем директорию для статических файлов (чтобы сервер отдавал стили css и скрипты js)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Указываем папку, где лежат HTML-шаблоны Jinja2
templates = Jinja2Templates(directory="templates")


# =====================================================================
# МАРШРУТЫ / ЭНДПОИНТЫ (API ROUTES)
# =====================================================================
@app.get("/")
async def index(request: Request):
    """Отображает главную страницу интерфейса пользователя"""
    return templates.TemplateResponse(request, "index.html", {})


@app.post("/detect")
@limiter.limit("10/minute")  # Защита: не более 10 отправленных картинок в минуту с одного IP
def detect(request: Request, file: UploadFile = File(...)):
    """Принимает файл изображения, прогоняет через ИИ, логирует и возвращает координаты рамок"""
    start_time = time.time()  # Фиксируем время начала обработки

    # Читаем бинарный контент загруженного файла напрямую из оперативной памяти
    contents = file.file.read() 
    
    # Конвертируем байты картинки в строку Base64, чтобы её можно было сохранить в БД и сразу вывести в HTML
    image_b64 = base64.b64encode(contents).decode("utf-8")
    image_url = f"data:image/jpeg;base64,{image_b64}"
    
    # Открываем изображение через библиотеку PIL для передачи в нейросеть YOLO
    image = Image.open(io.BytesIO(contents))

    # Вызываем метод детекции одежды, цвета и бренда
    detections, boxes_data = detector.detect(image)

    # Вычисляем время, затраченное на процессинг
    process_time = time.time() - start_time
    
    # Асинхронно/потоково пишем логи в SQLite базу данных
    history_manager.add(file.filename, image_url, detections, process_time)

    # Возвращаем JSON с результатами детекции и разметкой боксов
    return {"detections": detections, "boxes": boxes_data}


@app.get("/history")
def get_history():
    """Возвращает структурированный список всей истории сканирований в формате JSON"""
    return {"history": history_manager.get_all()}

@app.get("/stats")
def get_stats():
    """Возвращает агрегированные данные аналитики и счетчики классов/цветов в JSON"""
    return {"stats": history_manager.get_stats()}

@app.delete("/history")
def clear_history():
    """Удаляет всю историю и возвращает статус-сообщение об успешной очистке"""
    history_manager.clear()
    return {"message": "история очищена"}


# =====================================================================
# ЗАПУСК СЕРВЕРА
# =====================================================================
if __name__ == "__main__":
    import uvicorn
    # Получаем порт из переменных окружения хостинга (например, Railway) или ставим 8000 по умолчанию
    port = int(os.environ.get("PORT", 8000))
    # Если запуск происходит в облаке Railway, слушаем внешний интерфейс 0.0.0.0, иначе — локальный 127.0.0.1
    host = "0.0.0.0" if os.environ.get("RAILWAY_ENVIRONMENT") else "127.0.0.1"
    # Динамически определяем имя текущего запускаемого файла скрипта (например, 'main'),
    # чтобы избежать жесткой привязки к строке "main:app" на случай переименования файла.
    module_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    uvicorn.run(f"{module_name}:app", host=host, port=port, reload=False)
