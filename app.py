import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageOps

# Load  trained model
model = tf.keras.models.load_model('/content/drive/MyDrive/Burmese Handwritten Digit Recognition/Digit_model.h5')


# Title
st.title("Real-Time Handwritting Burmese Digit Recognizer")

# Start the webcam
st.write("Starting the webcam...")


# Initialize the camera
img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    # Convert the uploaded file to an OpenCV image.
    image = Image.open(img_file_buffer).convert('L')
    image = ImageOps.invert(image)  # Invert the image colors
    image = image.resize((28, 28))
    image = np.array(image)
    image = image / 255.0
    image = image.reshape(1, 28, 28, 1)

    prediction = model.predict(image)
    predicted_digit = np.argmax(prediction)

    # Display the image and prediction
    st.image(image.reshape(28, 28), caption=f'Predicted Digit: {predicted_digit}', width=150)
    st.write(f'Predicted Digit: {predicted_digit}')
        