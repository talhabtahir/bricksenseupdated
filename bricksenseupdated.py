import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Load the model once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('/kaggle/input/models-comp/keras/default/1/170kmodelv10_version_cam_1.keras')

model = load_model()

# Define class labels
class_labels = ["Normal", "Cracked", "Not a Wall"]

# Define a function for the processing and prediction
def process_and_predict_image(image):
    # Convert image to numpy array
    original_img = np.array(image)
    
    # Save original dimensions
    orig_height, orig_width, _ = original_img.shape

    # Preprocess the image for the model
    img_resized = cv2.resize(original_img, (224, 224))
    img_tensor = np.expand_dims(img_resized, axis=0) / 255.0
    preprocessed_img = img_tensor
    
    # Define a new model that outputs the conv2d_3 feature maps and the prediction
    custom_model = Model(inputs=model.inputs, 
                         outputs=(model.layers[10].output, model.layers[-1].output))  # `conv2d_3` and predictions

    # Get the conv2d_3 output and the predictions
    conv2d_3_output, pred_vec = custom_model.predict(preprocessed_img)

    # Print shape for debugging
    st.write("Shape of conv2d_3_output:", conv2d_3_output.shape)
    conv2d_3_output = np.squeeze(conv2d_3_output)  # Remove batch dimension

    # Check the shape after squeeze
    st.write("Shape after squeeze:", conv2d_3_output.shape)

    # Debugging: Check prediction vector
    st.write("Prediction vector:", pred_vec)

    # Ensure the shape of `pred_vec` is as expected
    if pred_vec.shape[-1] != len(class_labels):
        st.error(f"Unexpected shape of prediction vector: {pred_vec.shape}. Expected number of classes: {len(class_labels)}.")
        return None, None
    
    # Prediction for the image
    pred = np.argmax(pred_vec)
    st.write(f"Predicted index: {pred}")

    # Ensure the prediction index is within the valid range
    if pred < 0 or pred >= len(class_labels):
        st.error(f"Prediction index {pred} is out of range for class labels.")
        return None, None

    # Continue with heatmap processing...
    # (Your heatmap generation code goes here)

    # Get the predicted class name
    predicted_class = class_labels[pred]

    # (Return processed image and predicted class as before)
    return original_img, predicted_class

# Streamlit app layout
st.title("Image Prediction and Contour Detection")
st.write("Upload an image, and the model will predict its class and draw contours on the regions with highest activations.")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file).convert('RGB')
    
    # Display the original image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Process the image and get the result
    contoured_img, predicted_class = process_and_predict_image(image)

    # Check if the prediction was successful
    if contoured_img is not None and predicted_class is not None:
        # Display the result
        st.write(f"Predicted Class: {predicted_class}")
        st.image(contoured_img, caption='Image with Contours and Heatmap Overlay', use_column_width=True)
    else:
        st.error("An error occurred during image processing.")
