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
    return tf.keras.models.load_model('170kmodelv10_version_cam_1.keras')

model = load_model()

# Define class labels
class_labels = ["Normal", "Cracked", "Not a Wall"]

# Define a function for the processing and prediction
def process_and_predict_image(image):
    try:
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
        conv2d_3_output = np.squeeze(conv2d_3_output)  # (28, 28, 32) feature maps

        # Prediction for the image
        pred = np.argmax(pred_vec)
        
        # Resize the conv2d_3 output to match the input image size
        upsampled_conv2d_3_output = cv2.resize(conv2d_3_output, (orig_width, orig_height), interpolation=cv2.INTER_LINEAR)

        # Average all the filters from conv2d_3 to get a single activation map
        heat_map = np.mean(upsampled_conv2d_3_output, axis=-1)  # (224, 224)

        # Normalize the heatmap for better visualization
        heat_map = np.maximum(heat_map, 0)  # ReLU to eliminate negative values
        heat_map = heat_map / heat_map.max()  # Normalize to 0-1

        # Threshold the heatmap to get the regions with the highest activation
        threshold = 0.5  # Adjust this threshold if needed
        heat_map_thresh = np.uint8(255 * heat_map)
        _, thresh_map = cv2.threshold(heat_map_thresh, int(255 * threshold), 255, cv2.THRESH_BINARY)

        # Find contours in the thresholded heatmap
        contours, _ = cv2.findContours(thresh_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours on the original image
        contoured_img = original_img.copy()  # Copy original image
        cv2.drawContours(contoured_img, contours, -1, (0, 255, 0), 2)  # Draw green contours

        # Convert the heatmap to RGB for display
        heatmap_colored = np.uint8(255 * cm.jet(heat_map)[:, :, :3])
        
        # Convert heatmap and contoured images to PIL format for Streamlit
        heatmap_image = Image.fromarray(heatmap_colored)
        contoured_image = Image.fromarray(contoured_img)

        # Create an overlay image
        # Convert heatmap to RGBA
        heatmap_colored = np.concatenate([heatmap_colored, np.full((heat_map.shape[0], heat_map.shape[1], 1), 128, dtype=np.uint8)], axis=-1)
        heatmap_image = Image.fromarray(heatmap_colored)
        
        # Overlay heatmap on original image
        original_img_pil = Image.fromarray(original_img)
        overlay_img = Image.blend(original_img_pil, heatmap_image.convert("RGBA"), alpha=0.5)  # Adjust alpha as needed
        
        # Convert overlay image to PIL format for Streamlit
        overlay_image = overlay_img.convert("RGB")

        # Get the predicted class name
        predicted_class = class_labels[pred]

        return heatmap_image, contoured_image, overlay_image, predicted_class

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None, None, None, None


# Streamlit app layout
st.title("Image Prediction and Contour Detection")
st.write("Upload an image, and the model will predict its class, display the heatmap, and draw contours on the regions with highest activations.")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file).convert('RGB')
    
    # Display the original image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Process the image and get the results
    heatmap_img, contoured_img, predicted_class = process_and_predict_image(image)

    # Check if the prediction and images were processed successfully
    if contoured_img is not None and predicted_class is not None:
        # Display the predicted class
        st.write(f"Predicted Class: {predicted_class}")

        # Display the heatmap
        st.image(heatmap_img, caption='Heatmap', use_column_width=True)

        # Display the contoured image
        st.image(contoured_img, caption='Image with Contours', use_column_width=True)
    else:
        st.error("An error occurred during image processing.")
