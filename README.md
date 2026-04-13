```markdown
# LookCheck

### Автоматическая классификация одежды методами глубокого обучения

---

## 📌 Описание

LookCheck — программное решение для многоклассовой классификации предметов одежды на основе свёрточных нейронных сетей. Модель обучена распознавать 6 категорий: кофта, штаны, обувь, кепка, футболка.

## 🧠 Архитектура

Собственная CNN-модель, построенная на базе TensorFlow/Keras. Оптимизация под высокую скорость инференса при сохранении точности >94%.

## 📊 Ключевые показатели

- **Accuracy:** 0.942
- **Loss (Categorical Crossentropy):** 0.18
- **Параметров модели:** ~2.3 млн

## 🛠 Стек

`Python 3` `TensorFlow 2.x` `Keras` `NumPy` `OpenCV` `Matplotlib` `Scikit-learn`

## 🚀 Запуск

```bash
git clone https://github.com/username/LookCheck
cd LookCheck
pip install -r requirements.txt
python main.py --mode predict --source test_image.jpg
