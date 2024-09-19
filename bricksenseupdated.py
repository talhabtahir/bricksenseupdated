import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

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

    # Preprocess the image for the model
    img = cv2.resize(original_img, (224, 224))
    img_tensor = np.expand_dims(img, axis=0) / 255.0
    preprocessed_img = img_tensor
    
    # Define a new model that outputs the conv2d_3 feature maps and the prediction
    custom_model = Model(inputs=model.inputs, 
                         outputs=(model.layers[10].output, model.layers[-1].output))  # `conv2d_3` and predictions

    # Get the conv2d_3 output and the predictions
    conv2d_3_output, pred_vec = custom_model.predict(preprocessed_img)
    conv2d_3_output = np.squeeze(conv2d_3_output)  # 28x28x32 feature maps

    # Prediction for the image
    pred = np.argmax(pred_vec)

    # Resize the conv2d_3 output to match the input image size
    # Resizing back to original image size, not the resized input size
    upsampled_conv2d_3_output = cv2.resize(conv2d_3_output, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_LINEAR)  # (original width, original height)

    # Average all the filters from conv2d_3 to get a single activation map
    heat_map = np.mean(upsampled_conv2d_3_output, axis=-1)  # Take the mean of the 32 filters, resulting in (original height, original width)

    # Normalize the heatmap for better visualization
    heat_map = np.maximum(heat_map, 0)  # ReLU to eliminate negative values
    heat_map = heat_map / heat_map.max()  # Normalize to 0-1

    # Threshold the heatmap to get the regions with the highest activation
    threshold = 0.5  # You can adjust this threshold
    heat_map_thresh = np.uint8(255 * heat_map)  # Convert heatmap to 8-bit image
    _, thresh_map = cv2.threshold(heat_map_thresh, int(255 * threshold), 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded heatmap
    contours, _ = cv2.findContours(thresh_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image (without the heatmap overlay)
    contoured_img_only = original_img.copy()  # Copy original image
    cv2.drawContours(contoured_img_only, contours, -1, (0, 255, 0), 2)  # Draw green contours (lines)

    # Get the predicted class name
    predicted_class = class_labels[pred]

    return contoured_img_only, predicted_class

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
    contoured_img_only, predicted_class = process_and_predict_image(image)

    # Display the result
    st.write(f"Predicted Class: {predicted_class}")
    st.image(contoured_img_only, caption='Image with Contours', use_column_width=True)
