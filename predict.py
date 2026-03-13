import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model("pea_disease_model.keras")

img = image.load_img(r"C:\Users\darsh\Desktop\Pea_disease\dataset\Train\POWDER_MILDEW_LEAF\resized_3.jpg", target_size=(224,224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

prediction = model.predict(img_array)

classes = ['Downy Mildew','Healthy','Leaf Minner','Powdery Mildew']

predicted_index = np.argmax(prediction)
predicted_class = classes[predicted_index]
confidence = prediction[0][predicted_index] * 100

print("Detected Disease:", predicted_class)
print("Confidence:", round(confidence,2), "%")