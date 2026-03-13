import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Image preprocessing
train_data = ImageDataGenerator(rescale=1./255)
test_data = ImageDataGenerator(rescale=1./255)

# Load training images
train_generator = train_data.flow_from_directory(
    "dataset/Train",
    target_size=(224,224),
    batch_size=32,
    class_mode="categorical"
)

# Load testing images
test_generator = test_data.flow_from_directory(
    "dataset/Test",
    target_size=(224,224),
    batch_size=32,
    class_mode="categorical"
)

# Build CNN model
model = Sequential()

model.add(Conv2D(32,(3,3),activation='relu',input_shape=(224,224,3)))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dense(4,activation='softmax'))  # 4 classes

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)

# Evaluate model on test data
loss, accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss: {loss:.4f}")

# Save the model
model.save("pea_disease_model.keras  ")

print("Model training completed and saved.")