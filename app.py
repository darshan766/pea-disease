from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
import io
import tflite_runtime.interpreter as tflite

app = Flask(__name__)

# Load TFLite model
interpreter = tflite.Interpreter(model_path="pea_disease_model.tflite")
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

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

    img = image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]['index'])
    predicted_index = np.argmax(prediction)
    predicted_class = classes[predicted_index]
    confidence = round(float(prediction[0][predicted_index]) * 100, 2)

    return jsonify({
        "disease": predicted_class,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(debug=True)
