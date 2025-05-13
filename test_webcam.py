import cv2
import numpy as np
import tensorflow as tf
import os

# Load model and labels
model = load_model('asl_model.h5')
labels = np.load('labels.npy')

# Load test image
image_path = 'A:\University\BS-EE\Eightth semester\Deep Learning\Semester Project\ASL DEEP LEARNING PROJECT\A_test.jpg'  # Replace with your image path
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (64, 64)) / 255.0
img = np.reshape(img, (1, 64, 64, 1))

# Predict
prediction = model.predict(img)
predicted_class = np.argmax(prediction)
confidence = np.max(prediction)
predicted_letter = labels[predicted_class]

print(f"Predicted Letter: {predicted_letter}")
print(f"Confidence: {confidence:.2f}")