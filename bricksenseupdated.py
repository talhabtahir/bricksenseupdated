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
    layout="centered",
    menu_items={
        'Get Help': 'https://example.com/help',
        'Report a bug': 'https://example.com/bug',
        'About': 'Developed by BrickSense Team | © 2024'}
)

# # Custom CSS to hide specific Streamlit elements
# hide_github_style = """
# <style>
# /* Hide the top-right menu entirely (including GitHub links) */
# header {visibility: hidden;}
# footer {visibility: visible;}

# /* Alternatively, to hide just specific links */
# /* Target GitHub repo link and fork link directly by their classes or IDs */
# /* Uncomment and modify the lines below based on the inspection */

# /* a[href*="github.com/your-repo-name"] {display: none !important;} */

# /* .css-1q1n0ol a[aria-label="View source"] {display: none !important;} */

# /* iframe[src*="github.com"] {display: none !important;} */

# </style>
# """

# # Inject the custom CSS into the Streamlit app
# st.markdown(hide_github_style, unsafe_allow_html=True)


# # Helper function to convert image to base64
# def image_to_base64(img):
#     buffered = io.BytesIO()
#     img.save(buffered, format="PNG")
#     return base64.b64encode(buffered.getvalue()).decode()
# Display logo instead of header
imagelogo = Image.open("static/sidelogo2.png")
st.image(imagelogo, use_column_width=True, width=200)  # Update the path to your logo file
# # Use the helper function to encode the image
# encoded_logo = image_to_base64(imagelogo)

# # Center the logo using custom HTML and CSS
# st.markdown(
#     """
#     <div style='text-align: center;'>
#         <img src='data:image/png;base64,{}' width='600'>
#     </div>
#     """.format(encoded_logo),
#     unsafe_allow_html=True
# )



# Add space below the logo
st.write("")  # Creates a blank line
# st.write(" ")  # Creates an extra line for more space
# st.write(" ")  # Adjust the number of empty lines for desired spacing

# # Custom CSS to adjust the width of the selectbox
# st.markdown("""
#     <style>
#     .stSelectbox > div:first-child {
#         width: 100px !important;  /* Adjust this value to change the width */
#     }
#     </style>
#     """, unsafe_allow_html=True)
# Add a dropdown for model selection
# model_option = st.selectbox(    
#     "Choose a model for prediction:",
#     ("230kmodelv13_version_cam_3","230kmodelv12_version_cam_2" ,"230kmodelv11_version_cam_2", "170kmodelv10_version_cam_1", "v11model_cp_epoch_14", "v12model_cp_epoch_10")
# )

# Show selected model below the logo
# st.write(f"Selected Model: {model_option}")

# Sidebar navigation with icons
st.sidebar.image("static/sidelogo1.png", width=200, use_column_width=True)
st.sidebar.markdown("### ")
st.sidebar.markdown("### ")
st.sidebar.markdown("### ")

# #Model selection Toolbar
# st.sidebar.markdown("### Model Selection")
# model_option = st.sidebar.selectbox(
#     "Select the model to use:",
#     ("230kmodelv11_version_cam_2", "170kmodelv10_version_cam_1", "170kmodelv3_version_cam_1")
# )

@st.cache_resource
# For model comparison with sidebar selection
def load_model(model_name):
    try:
        model = tf.keras.models.load_model(f'{model_name}.keras')
        return model
    except Exception as e:
        st.error(f"Failed to load {model_name}: {e}")
        return None

# Load the selected model
model = load_model(model_option)
#__________________________________________________________________________________________________________________________________________________________________________________
# #For single model selection
# def load_model():
#     try:
#         model = tf.keras.models.load_model('230kmodelv11_version_cam_2.keras')
#         return model
#     except Exception as e:
#         st.error(f"Failed to load model: {e}")
#         return None

# model = load_model()
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


# Function to localize the crack and to make predictions using the TensorFlow model
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

        # Preprocess the image for the model
        img_resized = cv2.resize(original_img, (224, 224))
        img_tensor = np.expand_dims(img_resized, axis=0) / 255.0
        preprocessed_img = img_tensor
        
        # Define a new model that outputs the conv2d_3 feature maps and the prediction
        custom_model = Model(inputs=model.inputs, 
                             outputs=(model.layers[sensitivity].output, model.layers[-1].output))  # `conv2d_3` and predictions

        # Get the conv2d_3 output and the predictions
        conv2d_3_output, pred_vec = custom_model.predict(preprocessed_img)
        conv2d_3_output = np.squeeze(conv2d_3_output)  # (28, 28, 32) feature maps

        # Prediction for the image
        pred = np.argmax(pred_vec)
        
        # Resize the conv2d_3 output to match the input image size
        heat_map_resized = cv2.resize(conv2d_3_output, (orig_width, orig_height), interpolation=cv2.INTER_LINEAR)

        # Average all the filters from conv2d_3 to get a single activation map
        heat_map = np.mean(heat_map_resized, axis=-1)  # (orig_height, orig_width)

        # Normalize the heatmap for better visualization
        heat_map = np.maximum(heat_map, 0)  # ReLU to eliminate negative values
        heat_map = heat_map / heat_map.max()  # Normalize to 0-1

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
        # Convert heatmap-overlayed image to RGB for contour drawing
        heatmap_overlay_rgb = heatmap_overlay.convert("RGB")
        heatmap_overlay_rgb_np = np.array(heatmap_overlay_rgb)
        # heatmap_overlay_np = np.array(heatmap_overlay)
        cv2.drawContours(heatmap_overlay_rgb_np, contours, -1, (0, 0, 0), contour_thickness)  # Draw blue contours

        # Convert overlay image to PIL format
        overlay_img = Image.fromarray(heatmap_overlay_rgb_np)

        # Get the predicted class name
        class_labels = ["Normal", "Cracked", "Not a Wall"]
        predicted_class = class_labels[pred]

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
                "230kmodelv12_version_cam_2", 
                "230kmodelv11_version_cam_2",
                "v12model_cp_epoch_10",
                "v11model_cp_epoch_14",
                "170kmodelv10_version_cam_1",
                "230kmodelv13_version_cam_3"
            ]
        
            # Initialize list for storing model results
            model_results = []
        
            # Create an expander for sensitivity adjustment
            with st.expander("🔍 Sensitivity Settings"):
                # Add a slider for selecting the sensitivity dynamically
                sensitivity = st.slider(
                    "Adjust Detection Sensitivity (Higher values increase detection sensitivity)",
                    min_value=0,   # Minimum value for sensitivity
                    max_value=12,   # Maximum value for sensitivity
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
                _, _, _, _, _, overlay_img = import_and_predict(image, model=model, sensitivity=sensitivity)
                overlay_images.append(overlay_img)
        
            # Display overlay images for each model in two columns
            col1, col2, col3 = st.columns(3)
        
            with col1:
                st.image(overlay_images[0], caption=f"{model_names[0]}", use_column_width=True)
                st.image(overlay_images[2], caption=f"{model_names[2]}", use_column_width=True)
            
            with col2:
                st.image(overlay_images[1], caption=f"{model_names[1]}", use_column_width=True)
                st.image(overlay_images[3], caption=f"{model_names[3]}", use_column_width=True)
        
            with col3:
                st.image(overlay_images[4], caption=f"{model_names[4]}", use_column_width=True)
                st.image(overlay_images[5], caption=f"{model_names[5]}", use_column_width=True)
        
        except Exception as e:
            st.error(f"Error processing the uploaded image: {e}")
        # try:
        #     # Try to open the uploaded image using PIL
        #     image = Image.open(file)
        #     if image is None:
        #         raise ValueError("Uploaded file is not a valid image.")
            
        #     # Correct the orientation if necessary
        #     image = correct_orientation(image)

            
            
        #     # Perform prediction
        #     predictions, _, _, _, _, _  = import_and_predict(image)
        #     if predictions is not None:
        #         predicted_class = np.argmax(predictions)
        #         prediction_percentages = predictions[0] * 100

        #          # Display prediction result
        #         if predicted_class == 0:
        #             st.success(f"✅ This is a normal brick wall.")
        #         elif predicted_class == 1:
        #             st.error(f"❌ This wall is a cracked brick wall. ")
        #         elif predicted_class == 2:
        #             st.warning(f"⚠️ This is not a brick wall.")
        #         else:
        #             st.error(f"❓ Unknown prediction result: {predicted_class}")

        #         st.write(f"**Prediction Percentages:**")
        #         # Display predictions in one line
        #         st.markdown(f"""
        #             <div style="display: flex; justify-content: space-between; font-size: 14px; color: #e0e0e0; background-color: #808080; padding: 3px; border-radius: 9px;">
        #                 <div style="text-align: center; flex: 1;">🟢 <strong>Normal Wall:</strong> {prediction_percentages[0]:.2f}%</div>
        #                 <div style="text-align: center; flex: 1;">🔴 <strong>Cracked Wall:</strong> {prediction_percentages[1]:.2f}%</div>
        #                 <div style="text-align: center; flex: 1;">🟠 <strong>Not a Wall:</strong> {prediction_percentages[2]:.2f}%</div>
        #             </div>
        #         """, unsafe_allow_html=True)

        #         # st.write("")  # Creates a blank line
             
        #         # Create an expander for sensitivity adjustment
        #         with st.expander("🔍 Sensitivity Settings"):
        #             # Add a slider for selecting the sensitivity dynamically
        #             sensitivity = st.slider(
        #                 "Adjust Detection Sensitivity (Higher values increase detection sensitivity)",
        #                 min_value=0,   # Minimum value for sensitivity
        #                 max_value=12,   # Maximum value for sensitivity
        #                 value=9,       # Default value for sensitivity
        #                 step=1,        # Step for incremental changes
        #                 format="%.1f"    # Format to display sensitivity with one decimal
        #                                     )
        #         # layers name based on slider  
        #         layername= model.layers[sensitivity].name
        #         layershape= model.layers[sensitivity].output.shape
        #         # Display Layer info for selected model in two columns
        #         col1, col2 = st.columns(2)
        #         with col1:
        #             st.write(f"Layer Name: ",layername)
                
        #         with col2:
        #             st.write(f"Layer Shape: ",layershape)

        #         # Perform prediction again
        #         # predictions, image_with_border, contours_with_border, heatmap_image, contoured_image, overlay_img  = import_and_predict(image, sensitivity=sensitivity)

        #         # Define the models to compare
        #         model_names = [
        #         "230kmodelv12_version_cam_2", 
        #         "230kmodelv11_version_cam_2",
        #         "v12model_cp_epoch_10",
        #         "v11model_cp_epoch_14",
        #         "170kmodelv10_version_cam_1",
        #         "230kmodelv13_version_cam_3"
        #         ]
    
        #         overlay_images = []
    
        #         # Iterate over each model, perform prediction, and get overlay image
        #         for model_name in model_names:
        #             # Set the model to be used
        #             model = load_model_by_name(model_name)
                    
        #             # Perform prediction for the current model
        #             predictions, _, _, _, _, overlay_img = import_and_predict(image, model=model, sensitivity=sensitivity)
        #             overlay_images.append(overlay_img)
    
        #         # Display overlay images for each model in two columns
        #         col1, col2, col3= st.columns(3)
    
        #         with col1:
        #             st.image(overlay_images[0], caption=f"{model_names[0]}", use_column_width=True)
        #             st.image(overlay_images[2], caption=f"{model_names[2]}", use_column_width=True)
                
        #         with col2:
        #             st.image(overlay_images[1], caption=f"{model_names[1]}", use_column_width=True)
        #             st.image(overlay_images[3], caption=f"Model: {model_names[3]}", use_column_width=True)

        #         with col3:
        #             st.image(overlay_images[4], caption=f"{model_names[4]}", use_column_width=True)
        #             st.image(overlay_images[5], caption=f"{model_names[5]}", use_column_width=True)               

               
                
                               

        # except Exception as e:
        #     st.error(f"Error processing the uploaded image: {e}")

# Footer
# st.markdown("<div class='footer'>Developed with Streamlit & TensorFlow | © 2024 BrickSense</div>", unsafe_allow_html=True)

# footer_style = """
#     <style>
#     .footer {
#         position: fixed;
#         left: 0;
#         bottom: 0;
#         width: 100%;
#         background-color: white;
#         color: gray;
#         text-align: center;
#         font-size: small;
#         padding: 10px;
#     }
#     </style>
#     <div class='footer'>
#         Developed with Streamlit & TensorFlow | © 2024 BrickSense
#     </div>
# """

# st.markdown(footer_style, unsafe_allow_html=True)

import streamlit as st

st.markdown(
    """
    <div style="
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: white;
        color: gray;
        text-align: center;
        font-size: small;
        padding: 10px;
    ">
        Developed with Streamlit & TensorFlow | © 2024 BrickSense
    </div>
    """,
    unsafe_allow_html=True
)
