import tensorflow as tf
import os

print("Loading model...")
model = tf.keras.models.load_model("pea_disease_model.keras", compile=False)

print("Saving as SavedModel format...")
model.export("saved_model_temp")

print("Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_saved_model("saved_model_temp")
tflite_model = converter.convert()

with open("pea_disease_model.tflite", "wb") as f:
    f.write(tflite_model)

# Cleanup temp folder
import shutil
shutil.rmtree("saved_model_temp")

print(f"Done! TFLite model size: {len(tflite_model) / 1024 / 1024:.2f} MB")
