import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, ExifTags, ImageEnhance
import numpy as np
import cv2
from keras.models import Model
from streamlit_image_comparison import image_comparison
import math
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
# import io
# import base64

# Set the page configuration with favicon
st.set_page_config(
    page_title="Brick Detection",
    page_icon="static/brickicon8.png",  # Path to your favicon file
    layout="centered"
)


# Display logo instead of header
imagelogo = Image.open("static/sidelogo2.png")
st.image(imagelogo, use_container_width=True, width=200)  # Update the path to your logo file

# Add space below the logo
st.write("")  # Creates a blank line
# st.write(" ")  # Creates an extra line for more space
# st.write(" ")  # Adjust the number of empty lines for desired spacing


#__________________________________________________________________________________________________________________________________________________________________________________
#For single model selection
def load_model():
    try:
        model = tf.keras.models.load_model('230kmodelv11_version_cam_2.keras')
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model()
#__________________________________________________________________________________________________________________________________________________________________________________

# Sidebar for app information
st.sidebar.header("About This App")
st.sidebar.write("""This app uses a Convolutional Neural Network (CNN) model to detect brick walls and classify them as either normal, cracked, or not a wall. You can upload an image, and the app will analyze it to provide a prediction.""")
st.sidebar.write("""**Developed by:**  
Talha Bin Tahir  
**Email:** talhabtahir@gmail.com
""")

# Main area for image upload
file = st.file_uploader("Please upload an image of the brick wall", type=("jpg", "png", "jpeg", "bmp", "tiff", "webp"))

# Function to correct image orientation based on EXIF data
def correct_orientation(image):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = image._getexif()
        if exif is not None:
            orientation = exif.get(orientation, 1)
            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        pass
    return image
    
# Adding Canvas Background
def add_canvas(image, fill_color=(255, 255, 255)):
    """Automatically adjusts canvas size according to image size, with added padding and centers the image on the canvas."""
    # Get the original image size
    image_width, image_height = image.size
    
    # Calculate new canvas size with padding
    canvas_width = image_width + math.ceil(0.015 * image_width)
    canvas_height = image_height + math.ceil(0.07 * image_height)
    
    # Create a new image (canvas) with the calculated size
    canvas = Image.new("RGB", (canvas_width, canvas_height), fill_color)
    
    # Calculate the position to paste the image at the center of the canvas
    paste_position = (
        (canvas_width - image_width) // 2,
        (canvas_height - image_height) // 7
    )
    
    # Paste the original image onto the canvas
    canvas.paste(image, paste_position)
    
    return canvas

def import_and_predict(image_data, model=model, sensitivity=9):
    try:
        # Convert image to numpy array
        original_img = np.array(image_data)

        # Save original dimensions
        orig_height, orig_width, _ = original_img.shape

        # Calculate the maximum dimension of the original image
        max_dimension = max(orig_width, orig_height)

        # Set the scaling factor for contour line thickness based on the max dimension
        contour_thickness = max(2, int(max_dimension / 200))  # Adjust the divisor to control scaling

        # Check if the image has 4 channels (RGBA) and convert to RGB
        if original_img.shape[2] == 4:
            original_img = cv2.cvtColor(original_img, cv2.COLOR_RGBA2RGB)

        # Preprocess the image for the model
        img_resized = cv2.resize(original_img, (224, 224))
        img_tensor = np.expand_dims(img_resized, axis=0) / 255.0  # Normalize to [0, 1]

        # Ensure sensitivity index is within bounds
        if sensitivity < 0 or sensitivity >= len(model.layers):
            raise ValueError(f"Sensitivity index {sensitivity} is out of bounds for the model with {len(model.layers)} layers.")

        # Define a new model that outputs the desired layers
        custom_model = Model(inputs=model.inputs, 
                             outputs=(model.layers[sensitivity].output, model.layers[-1].output))

        # Get the conv layer output and predictions
        conv_output, pred_vec = custom_model.predict(img_tensor)

        # Debugging statements
        print("Conv output shape:", conv_output.shape)
        print("Prediction vector shape:", pred_vec.shape)

        # Check the shape of conv_output
        if len(conv_output.shape) not in {3, 4}:
            raise ValueError(f"Unexpected conv_output shape: {conv_output.shape}")

        # If 4D, squeeze the first dimension (batch size)
        if len(conv_output.shape) == 4:
            conv_output = np.squeeze(conv_output)  # Remove the batch dimension
            print("Shape after squeeze:", conv_output.shape)

        # Determine the heatmap generation based on output shape
        if conv_output.ndim == 3:
            heat_map = np.mean(conv_output, axis=-1)  # Average across the filters
        elif conv_output.ndim == 1:
            # Handle case where the output is a 1D array (e.g., shape could be (32,))
            heat_map = conv_output  # Use it directly

        # Resize the heatmap to match the original image dimensions
        heat_map_resized = cv2.resize(heat_map, (orig_width, orig_height), interpolation=cv2.INTER_LINEAR)

        # Normalize the heatmap for better visualization
        heat_map = np.maximum(heat_map_resized, 0)  # ReLU to eliminate negative values
        heat_map = heat_map / heat_map.max() if heat_map.max() > 0 else heat_map  # Avoid division by zero

        # Threshold the heatmap to get the regions with the highest activation
        threshold = 0.5  # Adjust this threshold if needed
        heat_map_thresh = np.uint8(255 * heat_map)
        _, thresh_map = cv2.threshold(heat_map_thresh, int(255 * threshold), 255, cv2.THRESH_BINARY)

        # Find contours in the thresholded heatmap
        contours, _ = cv2.findContours(thresh_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Convert heatmap to RGB for display
        heatmap_colored = np.uint8(255 * cm.jet(heat_map)[:, :, :3])

        # Convert heatmap to PIL format
        heatmap_image = Image.fromarray(heatmap_colored)

        # Create contoured image
        contoured_img = original_img.copy()  # Copy original image
        cv2.drawContours(contoured_img, contours, -1, (0, 0, 255), contour_thickness)  # Draw blue contours

        # Convert contoured image to PIL format
        contoured_image = Image.fromarray(contoured_img)

        # Overlay heatmap on original image
        heatmap_image_rgba = heatmap_image.convert("RGBA")
        original_img_pil = Image.fromarray(original_img).convert("RGBA")
        heatmap_overlay = Image.blend(original_img_pil, heatmap_image_rgba, alpha=0.5)

        # Draw contours on the heatmap-overlayed image
        heatmap_overlay_rgb = heatmap_overlay.convert("RGB")
        heatmap_overlay_rgb_np = np.array(heatmap_overlay_rgb)
        cv2.drawContours(heatmap_overlay_rgb_np, contours, -1, (0, 0, 0), contour_thickness)  # Draw contours

        # Convert overlay image to PIL format
        overlay_img = Image.fromarray(heatmap_overlay_rgb_np)

        # Get the predicted class name
        class_labels = ["Normal", "Cracked", "Not a Wall"]
        predicted_class = class_labels[np.argmax(pred_vec)]

        # Add white borders
        border_size = 10  # Set the border size
        image_with_border = add_white_border(image_data, border_size)
        contours_with_border = add_white_border(overlay_img, border_size)

        return pred_vec, image_with_border, contours_with_border, heatmap_image, contoured_image, overlay_img

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return None, None, None, None, None, None

# Function to load the model based on its name
def load_model_by_name(model_name):
    """Load the model based on the given model name."""
    model_path = f"{model_name}.keras"  # Adjust the path as needed
    model = tf.keras.models.load_model(model_path)
    return model

# Adds border to the image
def add_white_border(image, border_size):
    """Add a white border to the image."""
    return ImageOps.expand(image, border= border_size, fill=(255, 255, 255))

# Check if a file was uploaded
if file is None:
    st.info("Please upload an image file to start the detection.")
else:
    with st.spinner("Processing image..."):
        try:
            # Try to open the uploaded image using PIL
            image = Image.open(file)
            if image is None:
                raise ValueError("Uploaded file is not a valid image.")
            
            # Correct the orientation if necessary
            image = correct_orientation(image)
        
            # Define the models to compare
            model_names = [
                "230kmodelv11_version_cam_2",
                "230kmodelv12_version_cam_2",
                "230kmodelv13_version_cam_3",
                "230kmodelv141_version_cam_3",
                "230kmodelv142_version_cam_3",
                "230kmodelv143_version_cam_3",
                "230kmodelv151_version_cam_3",
                "230kmodelv17_version_cam_3"
                
            ]
        
            # Initialize list for storing model results
            model_results = []
        
            # Create an expander for sensitivity adjustment
            with st.expander("🔍 Sensitivity Settings"):
                # Add a slider for selecting the sensitivity dynamically
                sensitivity = st.slider(
                    "Adjust Detection Sensitivity (Higher values increase detection sensitivity)",
                    min_value=0,   # Minimum value for sensitivity
                    max_value=23,   # Maximum value for sensitivity
                    value=9,       # Default value for sensitivity
                    step=1,        # Step for incremental changes
                    format="%.1f"  # Format to display sensitivity with one decimal
                )
        
            # Iterate over each model, perform prediction, and get required information
            for model_name in model_names:
                # Set the model to be used
                model = load_model_by_name(model_name)
                
                # Perform prediction for the current model
                predictions, _, _, _, _, overlay_img = import_and_predict(image, model=model, sensitivity=sensitivity)
                
                # Calculate prediction percentages
                prediction_percentages = predictions[0] * 100
                
                # Determine predicted class
                predicted_class = np.argmax(predictions)
                if predicted_class == 0:
                    prediction_label = "Normal Wall"
                elif predicted_class == 1:
                    prediction_label = "Cracked Wall"
                elif predicted_class == 2:
                    prediction_label = "Not a Wall"
                else:
                    prediction_label = "Unknown"
                
                # Get the current layer name and shape based on sensitivity
                layer_name = model.layers[sensitivity].name
                layer_shape = model.layers[sensitivity].output.shape
                
                # Get total layers in the model
                total_layers = len(model.layers)
                
                # Append model information to the results list
                model_results.append({
                    "Model": model_name,
                    "Normal %": f"{prediction_percentages[0]:.2f}%",
                    "Cracked %": f"{prediction_percentages[1]:.2f}%",
                    "Not a Wall %": f"{prediction_percentages[2]:.2f}%",
                    "Prediction": prediction_label,
                    "Total Layers": total_layers,
                    "Current Layer": layer_name,
                    "Layer Shape": str(layer_shape)
                })
        
            # Convert the results into a DataFrame for display
            df = pd.DataFrame(model_results)
            
            # Insert a serial number (S.N.) column at the beginning
            df.insert(0, "S.N.", range(1, len(df) + 1))

            # Convert the DataFrame to HTML without the index
            html_table = df.to_html(index=False, classes='table table-bordered', escape=False)

                   
            # Render the table in Streamlit using markdown
            st.markdown(html_table, unsafe_allow_html=True)
            
        
            # Optionally, display overlay images for each model in columns
            overlay_images = []
            for model_name in model_names:
                model = load_model_by_name(model_name)
                if sensitivity >= len(model.layers):
                        # st.warning(f"Sensitivity value exceeds available layers for {model_name}. Maximum layer is {len(model.layers)-1}.")
                        continue  # Skip the model if the sensitivity exceeds available layers
                _, _, _, _, _, overlay_img = import_and_predict(image, model=model, sensitivity=sensitivity)
                overlay_images.append(overlay_img)
                
            # st.image(overlay_images[0], caption=f"{model_names[0]}", use_container_width=True)
            # # Display overlay images for each model in two columns
            # col1= st.columns(1)

            # Create columns dynamically based on the number of images/models
            cols = st.columns(4)  # Adjust the number of columns as needed (e.g., 4 for 4 models per row)
            
            for idx, overlay_img in enumerate(overlay_images):
                with cols[idx % 4]:  # Cycle through columns
                    st.image(overlay_img, caption=f"{model_names[idx]}", use_container_width=True)
                    
            
        except Exception as e:
            st.error(f"Error processing the uploaded image: {e}")
# Footer
# st.markdown("<div class='footer'>Developed with Streamlit & TensorFlow | © 2024 BrickSense</div>", unsafe_allow_html=True)
# Footer
footer = """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        background-color: #f5f5f5; /* Adjust background color to match the app theme */
        color: #333; /* Adjust text color for contrast */
        text-align: center;
        font-size: 0.9rem; /* Adjust font size for minimal appearance */
        margin: 0; /* Ensure no extra spacing */
        border-top: 1px solid #e0e0e0; /* Optional border for separation */
    }
    </style>
    <div class="footer">
        Developed with <span style="color: #e25555;">♥</span> using Streamlit & TensorFlow | © 2024 BrickSense
    </div>
"""
st.markdown(footer, unsafe_allow_html=True)
