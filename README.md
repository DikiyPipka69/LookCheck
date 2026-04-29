<div align="center">

<img src="https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/YOLO-v26-purple?style=for-the-badge"/>
<img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
<img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>

# 👗 LookChecker

**ИИ который определяет тип одежды на фотографии**

[🚀 Демо](#) • [📖 Документация](#установка) • [🐛 Баги](https://github.com/DikiyPipka69/project/issues)

</div>

---

## ✨ Возможности

- 🔍 **Определение одежды** — находит и классифицирует одежду на фото
- 📊 **Уверенность модели** — показывает процент уверенности с прогресс-баром
- 🕐 **История запросов** — сохраняет все предыдущие анализы
- 🌙 **Тёмная/светлая тема** — переключение одной кнопкой
- ⚡ **Быстрый анализ** — результат за доли секунды

---

## 🎯 Классы одежды

| Класс | Описание |
|-------|----------|
| 👕 T-shirt | Футболка |
| 👖 Trousers | Брюки |
| 👟 Shoes | Обувь |
| 🧥 Longsliva | Лонгслив |
| 🎩 Hat | Шапка/кепка |
| 👗 Dress | Платье |
| 🩳 Shorts | Шорты |
| 🧣 Skirt | Юбка |

---

## 🛠️ Технологии

- **YOLO v26** — детекция и классификация объектов
- **FastAPI** — бэкенд сервер
- **HTML/CSS/JS** — фронтенд
- **Python 3.12** — язык программирования

---

## 🚀 Установка

### 1. Клонируй репозиторий
```bash
git clone https://github.com/DikiyPipka69/project.git
cd project
```

### 2. Создай окружение
```bash
conda create -n ai_academy python=3.12
conda activate ai_academy
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

## 📁 Структура проекта
LookChecker/
├── static/
│   ├── style.css       # стили
│   └── script.js       # логика фронтенда
├── templates/
│   └── index.html      # главная страница
├── dataset/            # датасет (не включён в репо)
├── runs/               # веса модели после обучения
├── main.py             # FastAPI сервер
├── train.py            # обучение модели
├── requirements.txt
└── README.md
---

## 🎓 Как обучить модель самому

```bash
python train.py
```

Модель обучается на датасете в папке `dataset/`. После обучения веса сохраняются в `runs/detect/`.

---

## 📄 Лицензия

Проект распространяется под лицензией [MIT](LICENSE).

---

<div align="center">
Сделано с ❤️
</div>
