# ♻️ Smart Recycling Companion

A web application built using **Streamlit** and **Gradio** powered by PyTorch (MobileNetV3) to classify waste item images into four categories: **Plastic**, **Metal**, **Glass**, or **Paper**, accompanied by exact confidence percentages and disposal guidelines.

---

## 🚀 How to Run the Applications

### 1. Streamlit Web App (Recommended)
Run the following command in your terminal:

```bash
python -m streamlit run streamlit_app.py
```

- Open your browser at `http://localhost:8501`.
- Drag & drop or upload any waste item image (or click one of the quick test sample buttons).
- View the **detected material**, **confidence percentage**, **probability breakdown**, and **recycling tips**.

---

### 2. Gradio Web App
Run the following command in your terminal:

```bash
python gradio_app.py
```

- Open your browser at `http://localhost:7860`.
- Upload an image or select a sample image from the built-in gallery.
- Interactively view confidence percentages and recycling tips.

---

## 🛠️ CLI Prediction Script
To run single image predictions from the command line:

```bash
python predict.py --image data/dataset-resized/glass/glass1.jpg
```

---

## 📂 Project Structure

- [`streamlit_app.py`](streamlit_app.py): Interactive Streamlit web user interface.
- [`gradio_app.py`](gradio_app.py): Interactive Gradio web interface.
- [`predict.py`](predict.py): Command-line interface for running predictions on individual images.
- [`app.py`](app.py): FastAPI web server and REST API endpoint.
- [`recycling_model.pth`](recycling_model.pth): Trained PyTorch MobileNetV3 model weights.
- [`class_names.json`](class_names.json): Label mappings (`glass`, `metal`, `paper`, `plastic`).
