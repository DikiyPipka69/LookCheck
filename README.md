<div align="center">

<img src="https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/YOLO-v26-purple?style=for-the-badge"/>
<img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
<img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>

# 👕 LookCheck

**ИИ который определяет тип одежды на фотографии**

</div>

---

## ✨ Возможности

- 🔍 **Определение одежды** — находит и классифицирует одежду на фото
- 📊 **Уверенность модели** — показывает процент уверенности с прогресс-баром
- 🕐 **История запросов** — сохраняет все предыдущие анализы
- 🌙 **Тёмная/светлая тема** — переключение одной кнопкой
- ⚡ **Быстрый анализ** — результат за доли секунды
- 🎨 **Определение цвета** — определяет цвет найденной одежды
- 🏷️ **Распознавание бренда** — определяет бренд одежды
- 🛍️ **Поиск на маркетплейсах** — ссылки на Wildberries, Ozon, Яндекс Маркет
- 📈 **Статистика** — графики и аналитика всех запросов
- 🌍 **Мультиязычность** — интерфейс на русском и английском

---

## 🎯 Классы одежды

| Класс | Описание |
|-------|----------|
| 👕 T-shirt | Футболка |
| 👖 Trousers | Брюки |
| 🧥 Hoodie | Худи |
| 🥼 Longsliva | Лонгслив |
| 🎩 Hat | Шапка/кепка |
| 👚 Polo | Поло |
| 🩳 Shorts | Шорты |
| 🧣 Skirt | Юбка |

---

## 🛠️ Технологии

- **YOLO v26** — детекция и классификация объектов
- **FastAPI** — бэкенд сервер
- **HTML/CSS/JS** — фронтенд
- **Python 3.12** — язык программирования

---

## 📁 Структура проекта

```
LookCheck/
├── static/
│   ├── script.js      # логика фронтенда
│   └── style.css      # стили
├── templates/
│   └── index.html     # главная страница
├── runs/              # веса обученных моделей
├── main.py            # FastAPI сервер + ООП классы
├── train.py           # обучение модели одежды
├── train_brand.py     # обучение модели брендов
└── requirements.txt
```

---

## 🚀 Установка

### 1. Клонируй репозиторий
```bash
git clone https://github.com/DikiyPipka69/LookCheck.git
cd LookCheck
```

### 2. Создай окружение
```bash
conda create -n LookCheck python=3.12
conda activate LookCheck
```

### 3. Установи зависимости
```bash
pip install -r requirements.txt
```

### 4. Запусти сервер
```bash
python main.py
```

### 5. Открой браузер
http://127.0.0.1:8000

---

## 🎓 Как обучить модель самому

```bash
python train.py
```

Модель обучается на датасете в папке `dataset/`.
После обучения веса сохраняются в `runs/detect/`.

---

## 📄 Лицензия

Проект распространяется под лицензией [MIT](LICENSE).

---

<div align="center">
Сделано с ❤️