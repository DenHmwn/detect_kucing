import os
import numpy as np
from flask import Flask, render_template, request, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array  # Fix import
import uuid

# Load model
try:
    model = load_model('cat_model.h5')
    print("Model berhasil dimuat!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Setup Flask
app = Flask(__name__, template_folder='.')
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Buat folder upload jika belum ada
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    img_path = None
    confidence = None

    if request.method == 'POST':
        file = request.files.get('file')
        
        if file and file.filename != '':
            # Generate unique filename to avoid conflicts
            file_ext = os.path.splitext(file.filename)[1]
            unique_filename = str(uuid.uuid4()) + file_ext
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            try:
                # Save uploaded file
                file.save(filepath)
                print(f"File saved: {filepath}")

                if model is not None:
                    # Preprocess gambar dengan ukuran yang sama dengan training
                    img = load_img(filepath, target_size=(224, 224))  # Konsisten dengan model
                    img_array = img_to_array(img) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)

                    # Prediksi
                    result = model.predict(img_array, verbose=0)
                    confidence_score = float(result[0][0])
                    
                    # Threshold untuk klasifikasi (bisa disesuaikan)
                    threshold = 0.5
                    if confidence_score > threshold:
                        prediction = "KUCING TERDETEKSI!"
                        confidence = f"{confidence_score * 100:.1f}%"
                    else:
                        prediction = "BUKAN KUCING"
                        confidence = f"{(1 - confidence_score) * 100:.1f}%"
                    
                    # Path relatif untuk template
                    img_path = url_for('static', filename=f'uploads/{unique_filename}')
                    print(f"Prediction: {prediction} (Confidence: {confidence})")
                else:
                    prediction = "Model tidak tersedia"
                    img_path = url_for('static', filename=f'uploads/{unique_filename}')
                    
            except Exception as e:
                print(f"Error processing image: {e}")
                prediction = f"Error: {str(e)}"
        else:
            prediction = "Tidak ada file yang diupload"

    return render_template('index.html', prediction=prediction, img_path=img_path, confidence=confidence)

@app.errorhandler(413)
def too_large(e):
    return "File terlalu besar! Maksimal 16MB", 413

if __name__ == '__main__':
    print("Starting Cat Detector Server...")
    print("Upload folder:", app.config['UPLOAD_FOLDER'])
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max 16MB
    app.run(debug=True, host='0.0.0.0', port=5000)