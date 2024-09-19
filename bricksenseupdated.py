import streamlit as st
import cv2
import numpy as np
from PIL import Image
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
    
    # Visualize individual feature maps for debugging if shape is as expected
    if len(conv2d_3_output.shape) == 3:
        num_feature_maps = conv2d_3_output.shape[-1]
        st.write(f"Visualizing {num_feature_maps} feature maps:")
        fig, ax = plt.subplots(4, 8, figsize=(12, 6))
        for i in range(num_feature_maps):
            ax[i // 8, i % 8].imshow(conv2d_3_output[:, :, i], cmap='viridis')
            ax[i // 8, i % 8].axis('off')
        st.pyplot(fig)
    else:
        st.write("Unexpected feature map shape. Unable to visualize individual maps.")

    # Select some feature maps for averaging
    selected_feature_maps = conv2d_3_output[:, :, :3]  # Use the first 3 feature maps as an example

    # Average selected feature maps to get a single activation map
    heat_map = np.mean(selected_feature_maps, axis=-1)  # Take the mean, resulting in (28, 28)

    # Resize the heatmap to match the original image size
    heat_map_resized = cv2.resize(heat_map, (orig_width, orig_height), interpolation=cv2.INTER_LINEAR)

    # Normalize the heatmap for better visualization
    heat_map_resized = np.maximum(heat_map_resized, 0)  # ReLU to eliminate negative values
    heat_map_resized = heat_map_resized / heat_map_resized.max()  # Normalize to 0-1

    # Create a heatmap overlay using colormap
    heatmap_colored = np.uint8(255 * cm.jet(heat_map_resized)[:, :, :3])  # Use a colormap (e.g., jet)

    # Overlay the heatmap on the original image using alpha blending
    overlayed_img = cv2.addWeighted(original_img, 0.6, heatmap_colored, 0.4, 0)

    # Debugging: Display the heatmap without overlay
    st.image(heatmap_colored, caption="Debug: Heatmap without Overlay", use_column_width=True)

    # Threshold the heatmap to get the regions with the highest activation
    threshold = 0.5  # You can adjust this threshold
    heat_map_thresh = np.uint8(255 * heat_map_resized)  # Convert heatmap to 8-bit image
    _, thresh_map = cv2.threshold(heat_map_thresh, int(255 * threshold), 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded heatmap
    contours, _ = cv2.findContours(thresh_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image
    contoured_img = overlayed_img.copy()  # Copy image with heatmap
    cv2.drawContours(contoured_img, contours, -1, (0, 255, 0), 2)  # Draw green contours (lines)

    # Get the predicted class name
    predicted_class = class_labels[pred]

    return contoured_img, predicted_class

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

    # Display the result
    st.write(f"Predicted Class: {predicted_class}")
    st.image(contoured_img, caption='Image with Contours and Heatmap Overlay', use_column_width=True)
