import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image

# Loading the trained SVM model
svm_model = joblib.load('svm_dogs_vs_cats.pkl')

img_width, img_height = 150, 150

def classify_image(image):
    image = cv2.resize(image, (img_width, img_height))
    image_flat = image.flatten().reshape(1, -1)
    label = svm_model.predict(image_flat)
    return 'Dog' if label == 0 else 'Cat'

st.title("Dog vs Cat Classifier")
st.write("Upload an image and the classifier will predict whether it's a dog or a cat.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)


    st.image(image, channels="BGR", caption="Uploaded Image", use_column_width=True)


    label = classify_image(image)
    st.write(f"The classifier predicts: **{label}**")
