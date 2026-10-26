import os
import uuid
import numpy as np
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "emotion_model.keras")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

CLASS_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def prepare_image(image_path):
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image).convert("L").resize((48, 48))
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.expand_dims(array, axis=(0, -1))
    return array


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    if request.method == "POST":
        uploaded = request.files.get("image")
        if not uploaded or uploaded.filename == "":
            error = "Please select an image file."
        elif not allowed_file(uploaded.filename):
            error = "Unsupported image type. Use PNG, JPG, JPEG, or BMP."
        elif not os.path.exists(MODEL_PATH):
            error = "Trained model not found. Follow the README instructions to train the model first."
        else:
            filename = f"{uuid.uuid4().hex[:8]}_{secure_filename(uploaded.filename)}"
            image_path = os.path.join(UPLOAD_DIR, filename)
            uploaded.save(image_path)
            try:
                model = load_model(MODEL_PATH)
                image_array = prepare_image(image_path)
                probabilities = model.predict(image_array, verbose=0)[0]
                best_index = int(np.argmax(probabilities))
                prediction = {
                    "label": CLASS_NAMES[best_index],
                    "confidence": round(float(probabilities[best_index]) * 100, 2),
                    "filename": filename,
                }
            except Exception as exc:
                error = f"Could not classify the image: {exc}"
            finally:
                try:
                    os.remove(image_path)
                except OSError:
                    pass

    return render_template("index.html", prediction=prediction, error=error)


if __name__ == "__main__":
    app.run(debug=True)
