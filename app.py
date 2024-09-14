import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
import requests
from streamlit_lottie import st_lottie

# Set the page title and expand the layout
st.set_page_config(page_title="growSmart", page_icon=":herb:", layout="wide")

# Function to load Lottie animation
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_coding = load_lottieurl("https://lottie.host/b9247bc8-f537-4cc0-9ad7-a2b3121b2274/gWtNiRf3VU.json")

# Display app title
st.title('growSmart🌿')
st.markdown('<p style="font-size: 14px; color: grey;">Detect. Prevent. Thrive.</p>', unsafe_allow_html=True)

# Add introductory text
with st.container():
    st.write("---")
    left_column, right_column = st.columns(2)
    with left_column:
        st.write("""
        ### ABOUT
                 
growSmart is an AI-powered platform designed to assist farmers in making informed decisions about their plants. Our innovative plant disease detector enables early identification of plant diseases.

Simply upload an image of a plant leaf, and we will help you pinpoint any potential issues. Our AI-driven tool is crafted to ensure your plants stay healthy and thrive.
        """)
    with right_column:
        st_lottie(lottie_coding, height=300, key='coding')

# Load the pre-trained model
model_path = 'trained_model/plant_disease_prediction_model.h5'
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    st.error(f"Error loading model: {e}")

# Load class indices
class_indices_path = 'class_indices.json'
try:
    with open(class_indices_path, 'r') as f:
        class_indices = json.load(f)
except Exception as e:
    st.error(f"Error loading class indices: {e}")

# Function to load and preprocess the image
def load_and_preprocess_image(image, target_size=(224, 224)):
    """
    This function takes an uploaded image file, processes it, and returns a numpy array 
    ready for model prediction.
    """
    # Ensure the image is in RGB format (3 channels) and resize it
    image = image.convert('RGB')
    
    # Resize the image to match the input size expected by the model
    image = image.resize(target_size)
    
    # Convert the image to a numpy array
    img_array = np.array(image)
    
    # Add an extra dimension to match the model's input shape (batch_size, width, height, channels)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize pixel values (0-255) to the range [0, 1] as expected by the model
    img_array = img_array.astype('float32') / 255.0
    
    return img_array

# Function to predict the class of an image
def predict_image_class(model, image, class_indices):
    """
    This function takes a pre-trained model, a preprocessed image, and a dictionary of 
    class indices, and returns the predicted class name.
    """
    # Preprocess the image using the above function
    preprocessed_img = load_and_preprocess_image(image)
    
    # Make predictions using the pre-trained model
    predictions = model.predict(preprocessed_img)
    
    # Get the predicted class index (the one with the highest probability)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    
    # Convert the class index to the corresponding class name using class_indices
    predicted_class_name = class_indices[str(predicted_class_index)]
    
    return predicted_class_name

# Image uploader
uploaded_image = st.file_uploader("Upload an image of a plant leaf...", type=["jpg", "jpeg", "png"])

# Display the uploaded image and make predictions
if uploaded_image is not None:
    # Load the image
    image = Image.open(uploaded_image)
    
    # Display the uploaded image on the left
    col1, col2 = st.columns(2)
    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img, caption="Uploaded Image", use_column_width=True)

    # Make predictions and show results on the right
    with col2:
        if st.button('Check for Diseases'):
            try:
                # Predict the class of the uploaded image
                predicted_class_name = predict_image_class(model, image, class_indices)
                st.success(f'Prediction: {predicted_class_name}')
            except Exception as e:
                st.error(f"Prediction error: {e}")

            # Fetch detailed disease information from the disease API
            api_url_disease = "http://40.117.96.109:8002/disease"
            api_data_disease = {"q": predicted_class_name}
            response_disease = requests.post(api_url_disease, json=api_data_disease)

            if response_disease.status_code == 200:
                data_disease = response_disease.json()
                if "response" in data_disease:
                    result = data_disease["response"]
                    st.success(f'Description: {result}')
                else:
                    st.error(f"Failed to retrieve description for {predicted_class_name}: Key not found in response")
            else:
                st.error("Failed to retrieve description from API: Status code indicates failure")

            # Chat functionality for follow-up questions
            user_question = st.text_input("Ask a follow-up question about the disease:")
            if st.button('Ask the Chatbot'):
                if user_question:
                    api_url_chat = "http://40.117.96.109:8002/chat"
                    api_data_chat = {"q": user_question}
                    response_chat = requests.post(api_url_chat, json=api_data_chat)

                    if response_chat.status_code == 200:
                        data_chat = response_chat.json()
                        if "response" in data_chat:
                            chat_result = data_chat["response"]
                            st.success(f'Chatbot: {chat_result}')
                        else:
                            st.error("Failed to retrieve response from chatbot.")
                    else:
                        st.error("Failed to connect to chatbot API.")
