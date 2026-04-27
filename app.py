import gradio as gr
from ultralytics import YOLO

model = YOLO("runs/detect/train16/weights/best.pt")

def detect(image_path):                # Основная функция кода непосредственно для распознавания
    if image_path is None:                # Проверка на наличие фото
        return "❌ Загрузите фото"
    results = model(image_path)            # Распознование
    output = ""
    for r in results:                       # Обработка результатов
        for box in r.boxes:
            class_name = model.names[int(box.cls[0])]        # Определение названия предмета
            confidence = float(box.conf[0])                     # Расчитывание уверенности
            output += f"👗 {class_name}: {confidence:.0%}\n"       # Перевод уверенности в проценты
    return output if output else "❌ Ничего не найдено"



                             # Дизайн сайта
css = """
.gradio-container {
    background-color: #0d0d0d !important;
    max-width: 700px !important;
    margin: auto !important;
}
#title {
    text-align: center;
    color: #a855f7;
    font-size: 2.5em;
    font-weight: bold;
    margin: 20px 0 5px 0;
}
#subtitle {
    text-align: center;
    color: #6b7280;
    font-size: 1em;
    margin-bottom: 30px;
}
#upload {
    border: 2px dashed #7c3aed !important;
    border-radius: 16px !important;
    background: #1a1a2e !important;
}
#upload {
    border: 2px dashed #7c3aed !important;
    border-radius: 16px !important;
    background: rgba(26, 26, 46, 0.7) !important; /* полупрозрачный */
    backdrop-filter: blur(4px) !important; /* опционально: размытие подложки */
    transition: all 0.3s ease !important;
}
#upload:hover {
    background: rgba(106, 90, 205, 0.5) !important; /* светлый полупрозрачный слой */
    border-color: #a855f7 !important; /* опционально: подсветка границы */
}
#result {
    background: #1a1a2e !important;
    border: 1px solid #7c3aed !important;
    border-radius: 16px !important;
    color: #e0e0e0 !important;
    font-size: 1.2em !important;
    text-align: center !important;
}
#btn {
    background: linear-gradient(135deg, #7c3aed, #a855f7) !important;
    border: none !important;
    color: white !important;
    font-weight: bold !important;
    border-radius: 10px !important;
    font-size: 1em !important;
}
#btn:hover {
    background: linear-gradient(135deg, #6d28d9, #9333ea) !important;
}
"""

with gr.Blocks(css=css, theme=gr.themes.Base()) as demo:                    # Применение css
    gr.HTML("<div id='title'>👕 LookCheck</div>")                                           # Заголовки
    gr.HTML("<div id='subtitle'>Загрузите фото одежды — модель определит тип</div>")

    with gr.Row():                         # Поле для загрузки фото
        image_input = gr.Image(
            type="filepath",
            label="",
            elem_id="upload",
            sources=["upload"],  # Только загрузка, без камеры и буфера
            show_label=False
        )

    with gr.Row():                                                 # Кнопка
        btn = gr.Button("🔍 Определить", elem_id="btn")

    with gr.Row():                    # Поле вывода
        result_output = gr.Text(
            label="",
            elem_id="result",
            show_label=False,
            placeholder="Результат появится здесь..."
        )

    btn.click(fn=detect, inputs=image_input, outputs=result_output)        # Связывание кнопки с функцией

if __name__ == "__main__":             # Стартуем
    demo.launch()















































