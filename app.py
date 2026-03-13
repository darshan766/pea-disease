from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("pea_disease_model.keras")

# Class labels (must match training folder order)
classes = ['Downy Mildew', 'Healthy', 'Leaf Minner', 'Powdery Mildew']

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    image = Image.open(io.BytesIO(file.read())).convert("RGB")

    # Resize and preprocess
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_class = classes[predicted_index]
    confidence = round(float(prediction[0][predicted_index]) * 100, 2)

    return jsonify({
        "disease": predicted_class,
        "confidence": confidence
    })

if __name__ == "__main__":
    from pyngrok import ngrok
    public_url = ngrok.connect(5000)
    print(f"\n Public URL: {public_url}\n")
    app.run(debug=False)
