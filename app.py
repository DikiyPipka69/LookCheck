import gradio as gr
from ultralytics import YOLO

model = YOLO("runs/detect/train13/weights/best.pt")

def detect(image_path):
    results = model(image_path)
    output = ""
    for r in results:
        for box in r.boxes:
            class_name = model.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            output += f"👗 {class_name}: {confidence:.0%}\n"
    return output if output else "❌ Ничего не найдено"

css = """
* {
    font-family: 'Segoe UI', sans-serif;
}
body, .gradio-container {
    background-color: #0d0d0d !important;
    color: #e0e0e0 !important;
}
.gradio-container {
    max-width: 800px !important;
    margin: auto !important;
}
h1 {
    color: #a855f7 !important;
    font-size: 2.5em !important;
    text-align: center !important;
    margin-bottom: 0.2em !important;
}
.description {
    text-align: center !important;
    color: #9ca3af !important;
    margin-bottom: 1.5em !important;
}
button.primary {
    background: linear-gradient(135deg, #7c3aed, #a855f7) !important;
    border: none !important;
    color: white !important;
    font-weight: bold !important;
    border-radius: 8px !important;
}
button.primary:hover {
    background: linear-gradient(135deg, #6d28d9, #9333ea) !important;
}
.block {
    background-color: #1a1a2e !important;
    border: 1px solid #7c3aed !important;
    border-radius: 12px !important;
}
textarea, .output-text {
    background-color: #12122a !important;
    color: #e0e0e0 !important;
    border: 1px solid #7c3aed !important;
    border-radius: 8px !important;
    font-size: 1.1em !important;
}
"""

demo = gr.Interface(
    fn=detect,
    inputs=gr.Image(type="filepath", label="📷 Загрузи фото одежды"),
    outputs=gr.Text(label="🔍 Результат"),
    title="LookChecker",
    description="Загрузи фото — модель определит тип одежды",
    css=css,
    theme=gr.themes.Base(
        primary_hue="purple",
        neutral_hue="slate",
    )
)

if __name__ == "__main__":
    demo.launch()














































































