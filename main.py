import streamlit as st
import numpy as np
from PIL import Image
import uuid
import os
from inference import preprocess_image, predict
from model import load_model
from db import insert_user_input

# Load model
W1, b1, W2, b2 = load_model("model.pkl")

# Folder to save user-labeled data
UPLOAD_FOLDER = "user_data"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

st.title("Handwritten Digit Recognition")

uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    image = image.resize((28, 28)) 
    st.image(image, caption="Uploaded Image", width=150)

    try:
        # Preprocess and predict
        X = preprocess_image(uploaded_file)
        pred, probs = predict(X, W1, b1, W2, b2)

        st.markdown(f"Model Prediction: `{pred}`")

        # User feedback section
        st.markdown("What digit do you think it is?")
        user_label = st.number_input("Enter the correct digit", min_value=0, max_value=9, step=1)

        if st.button("Submit Correct Label"):
            insert_user_input(user_label, image)
            st.success("Thank you! Your label has been saved.")

    except Exception as e:
        st.error(f"Error: {e}")
