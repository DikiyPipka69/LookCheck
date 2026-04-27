import gradio as gr
from ultralytics import YOLO

model = YOLO("runs/detect/train13/weights/best.pt")

def detect(image_path):
    if image_path is None:
        return "❌ Загрузите фото"
    results = model(image_path)
    output = ""
    for r in results:
        for box in r.boxes:
            class_name = model.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            output += f"👗 {class_name}: {confidence:.0%}\n"
    return output if output else "❌ Ничего не найдено"

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

with gr.Blocks(css=css, theme=gr.themes.Base()) as demo:
    gr.HTML("<div id='title'>👕 LookChecker</div>")
    gr.HTML("<div id='subtitle'>Загрузите фото одежды — модель определит тип</div>")

    with gr.Row():
        image_input = gr.Image(
            type="filepath",
            label="",
            elem_id="upload",
            sources=["upload"],  # только загрузка, без камеры и буфера
            show_label=False
        )

    with gr.Row():
        btn = gr.Button("🔍 Определить", elem_id="btn")

    with gr.Row():
        result_output = gr.Text(
            label="",
            elem_id="result",
            show_label=False,
            placeholder="Результат появится здесь..."
        )

    btn.click(fn=detect, inputs=image_input, outputs=result_output)

if __name__ == "__main__":
    demo.launch()














































































