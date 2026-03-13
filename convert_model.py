import tensorflow as tf

model = tf.keras.models.load_model("pea_disease_model.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("pea_disease_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Model converted to TFLite successfully.")
print(f"TFLite model size: {len(tflite_model) / 1024 / 1024:.2f} MB")
