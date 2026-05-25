import os
import numpy as np
import gradio as gr

from tensorflow.keras.models import load_model
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "cat_model.h5")

try:
    model = load_model(MODEL_PATH)
    print("✅ Model berhasil dimuat!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None


def predict_cat(image):
    if image is None:
        return "❌ Silakan upload gambar terlebih dahulu."

    if model is None:
        return "❌ Model tidak tersedia atau gagal dimuat."

    try:
        # Gradio akan mengirim gambar dalam bentuk PIL Image
        image = image.convert("RGB")
        image = image.resize((224, 224))

        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        result = model.predict(img_array, verbose=0)
        confidence_score = float(result[0][0])

        threshold = 0.5

        # Sesuaikan dengan label saat training:
        # Jika saat training: cat = 0 dan non_cat = 1, pakai logika ini.
        if confidence_score < threshold:
            confidence = (1 - confidence_score) * 100
            return f"✅ KUCING TERDETEKSI!\nConfidence: {confidence:.1f}%"
        else:
            confidence = confidence_score * 100
            return f"❌ BUKAN KUCING\nConfidence: {confidence:.1f}%"

    except Exception as e:
        return f"❌ Error saat memproses gambar: {str(e)}"


demo = gr.Interface(
    fn=predict_cat,
    inputs=gr.Image(type="pil", label="Upload gambar"),
    outputs=gr.Textbox(label="Hasil Prediksi"),
    title="🐱 Cat Detector AI",
    description="Upload gambar untuk mendeteksi apakah gambar tersebut kucing atau bukan.",
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()