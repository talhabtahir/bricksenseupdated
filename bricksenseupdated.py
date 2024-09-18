import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, ExifTags, ImageEnhance
import numpy as np
import cv2
from keras.models import Model
from streamlit_image_comparison import image_comparison
import math

# Set the page configuration with favicon
st.set_page_config(
    page_title="Brick Detection",
    page_icon="static/brickicon8.png",  # Path to your favicon file
    layout="centered"
)

# Custom CSS for additional styling
st.markdown(
    """
    <link rel="icon" href="static/brickicon8.png" type="image/x-icon">
    <style>
        .reportview-container {
            background-color: #f7f9fc;
            padding-top: 20px;
        }
        .sidebar .sidebar-content {
            background-color: #f7f9fc;
        }
        .main-header {
            color: #ff6347;
            text-align: center;
        }
        .footer {
            text-align: center;
            padding: 10px;
            font-size: small;
            color: #666;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Display logo instead of header
imagelogo = Image.open("static/sidelogo.png")
st.image(imagelogo, use_column_width=True, width=150)  # Update the path to your logo file

# Add space below the logo
st.write("")  # Creates a blank line
st.write(" ")  # Creates an extra line for more space
st.write(" ")  # Adjust the number of empty lines for desired spacing

# Sidebar navigation with icons
st.sidebar.image("static/sidelogo.png", width=200, use_column_width=True)
st.sidebar.markdown("### ")
st.sidebar.markdown("### ")
st.sidebar.markdown("### ")

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('170kmodelv10_version_cam_1.keras')
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model()

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

def apply_thresholding_and_morphology(heatmap):
    # Convert heatmap to 8-bit
    heatmap_8bit = np.uint8(255 * heatmap)

    # Apply thresholding
    _, binary_map = cv2.threshold(heatmap_8bit, 127, 255, cv2.THRESH_BINARY)

    # Define kernel for morphological operations
    kernel = np.ones((3, 3), np.uint8)

    # Dilation and Erosion
    dilated_map = cv2.dilate(binary_map, kernel, iterations=2)
    eroded_map = cv2.erode(dilated_map, kernel, iterations=1)

    return binary_map, dilated_map, eroded_map
  
# Function to localize the crack and to make predictions using the TensorFlow model
def import_and_predict(image_data, model, sensitivity=11):
    try:
        original_size = image_data.size
        original_width, original_height = original_size
        size = (224, 224)

        image_resized = image_data.convert("RGB")
        image_resized = ImageOps.fit(image_resized, size, Image.LANCZOS)
        img = np.asarray(image_resized).astype(np.float32) / 255.0
        img_reshape = img[np.newaxis, ...]

        custom_model = Model(inputs=model.inputs, 
                             outputs=(model.layers[sensitivity].output, model.layers[-1].output))
        layer_output, pred_vec = custom_model.predict(img_reshape)

        pred = np.argmax(pred_vec)
        layer_output = np.squeeze(layer_output)
        heat_map = np.mean(layer_output, axis=-1)

        heat_map = np.maximum(heat_map, 0)
        heat_map /= np.max(heat_map)

        heatmap_resized = cv2.resize(heat_map, size, interpolation=cv2.INTER_LINEAR)

        binary_map, dilated_map, eroded_map = apply_thresholding_and_morphology(heatmap_resized)

        original_img_np = np.array(image_data)

        if len(original_img_np.shape) == 2:
            original_img_np = cv2.cvtColor(original_img_np, cv2.COLOR_GRAY2RGB)

        original_img_bgr = cv2.cvtColor(original_img_np, cv2.COLOR_RGB2BGR)

        # Ensure dilated_map has the same size and channels as original_img_bgr
        dilated_map_rgb = cv2.cvtColor(dilated_map, cv2.COLOR_GRAY2RGB)
        dilated_map_rgb_resized = cv2.resize(dilated_map_rgb, (original_img_bgr.shape[1], original_img_bgr.shape[0]))

        overlay_img_bgr = cv2.addWeighted(original_img_bgr, 0.7, dilated_map_rgb_resized, 0.3, 0)

        contours_img_rgb = cv2.cvtColor(overlay_img_bgr, cv2.COLOR_BGR2RGB)
        contours_pil2 = Image.fromarray(contours_img_rgb)

        enhancer = ImageEnhance.Brightness(contours_pil2)
        contours_pil = enhancer.enhance(0.8)

        return pred_vec, contours_pil2

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return None, None
        
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

            
            
            # Perform prediction
            pred_vec, contours_pil2 = import_and_predict(image, model)
            
            if pred_vec is not None:
                predicted_class = np.argmax(pred_vec)
                prediction_percentages = pred_vec[0] * 100

                 # Display prediction result
                if predicted_class == 0:
                    st.success(f"✅ This is a normal brick wall.")
                elif predicted_class == 1:
                    st.error(f"❌ This wall is a cracked brick wall. ")
                elif predicted_class == 2:
                    st.warning(f"⚠️ This is not a brick wall.")
                else:
                    st.error(f"❓ Unknown prediction result: {predicted_class}")

                st.write(f"**Prediction Percentages:**")
                # Display predictions in one line
                st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; font-size: 14px; color: #e0e0e0; background-color: #808080; padding: 3px; border-radius: 9px;">
                        <div style="text-align: center; flex: 1;">🟢 <strong>Normal Wall:</strong> {prediction_percentages[0]:.2f}%</div>
                        <div style="text-align: center; flex: 1;">🔴 <strong>Cracked Wall:</strong> {prediction_percentages[1]:.2f}%</div>
                        <div style="text-align: center; flex: 1;">🟠 <strong>Not a Wall:</strong> {prediction_percentages[2]:.2f}%</div>
                    </div>
                """, unsafe_allow_html=True)

                st.write("")  # Creates a blank line

                # st.write("")  # Creates a blank line

                # Create an expander for sensitivity adjustment
                with st.expander("🔍 Sensitivity Settings"):
                    # Add a slider for selecting the sensitivity dynamically
                    sensitivity = st.slider(
                        "Adjust Detection Sensitivity (Higher values increase detection sensitivity)",
                        min_value=1,   # Minimum value for sensitivity
                        max_value=12,   # Maximum value for sensitivity
                        value=11,       # Default value for sensitivity
                        step=1,        # Step for incremental changes
                        format="%.1f"    # Format to display sensitivity with one decimal
                                            )
                # Perform prediction again
                pred_vec, contours_pil2 = import_and_predict(image, model, sensitivity=sensitivity)

                # Display the uploaded image and the contours side by side
                col1, col2 = st.columns(2)
    
                with col1:
                    st.image(image, caption="Uploaded Image", use_column_width=True)
                with col2:
                    if predicted_class == 1:
                        st.image(contours_pil2, caption="Cracks Localization", use_column_width=True)
                    elif predicted_class == 0:
                        st.image(image, caption="No cracks detected", use_column_width=True)
                    else:
                        st.image(image, caption="No wall detected", use_column_width=True)



               
                
                image_with_border = add_canvas(image_with_border)
                contours_with_border = add_canvas(contours_with_border)               
                # st.write(f"Normal Wall: {prediction_percentages[0]:.2f}%")
                # st.write(f"Cracked Wall: {prediction_percentages[1]:.2f}%")
                # st.write(f"Not a Wall: {prediction_percentages[2]:.2f}%")
                st.write("")  # Creates a blank line
                if st.checkbox("Original vs Cracked Slider"):
                        # Conditionally display image comparison
                        if predicted_class == 1:
                            image_comparison(
                                img1=image_with_border, 
                                img2=contours_with_border,
                                label1="Uploaded Image",
                                label2="Cracks Localization",
                                show_labels=False,
                                # width=670,
                                make_responsive=True,
                                # in_memory=True
                            )
                        else:
                           image_comparison(
                                img1=image_with_border, 
                                img2=image_with_border,
                                label1="Uploaded Image",
                                label2="Cracks Localization",
                                show_labels=False,
                                # width=670,
                                make_responsive=True,
                                # in_memory=True
                            )

                # # Conditionally display image comparison
                # if predicted_class == 1:
                #     st.markdown(
                #         """
                #         <div style='display: flex; justify-content: center; align-items: center;'>
                #         """, 
                #         unsafe_allow_html=True
                #     )
                #     image_comparison(
                #         img1=image_with_border, 
                #         img2=contours_with_border,
                #         label1="Uploaded Image",
                #         label2="Cracks Localization",
                #         show_labels=False,
                #         # width=670,
                #         make_responsive=True,
                #         in_memory=True
                #     )
                #     st.markdown("</div>", unsafe_allow_html=True)
                # else:
                #     st.markdown(
                #         """
                #         <div style='display: flex; justify-content: center; align-items: center;'>
                #         """, 
                #         unsafe_allow_html=True
                #     )
                #     image_comparison(
                #         img1=image_with_border, 
                #         img2=image_with_border,
                #         label1="Uploaded Image",
                #         label2="Cracks Localization",
                #         show_labels=False,
                #         # width=670,
                #         make_responsive=True,
                #         in_memory=True
                #     )
                #     st.markdown("</div>", unsafe_allow_html=True)

                # with st.expander("Original vs Cracked Slider"):
                #     # Define a maximum width for the images based on a rough estimate or browser inspection
                #     max_width = 600  # Adjust this based on what you see in browser inspection
                
                #     # Resize image accordingly to fit within the expander's width
                #     img1_resized = image_with_border.resize((max_width, int(image_with_border.height * (max_width / image_with_border.width))))
                #     img2_resized = contours_with_border.resize((max_width, int(contours_with_border.height * (max_width / contours_with_border.width))))
                
                #     # Conditionally display image comparison
                #     if predicted_class == 1:
                #         image_comparison(
                #             img1=img1_resized, 
                #             img2=img2_resized,
                #             label1="Uploaded Image",
                #             label2="Cracks Localization",
                #             show_labels=False
                #         )
                #     else:
                #         image_comparison(
                #             img1=img1_resized, 
                #             img2=img1_resized,
                #             label1="Uploaded Image",
                #             label2="Cracks Localization",
                #             show_labels=False
                #         )

        except Exception as e:
            st.error(f"Error processing the uploaded image: {e}")

# Footer
st.markdown("<div class='footer'>Developed with Streamlit & TensorFlow | © 2024 BrickSense</div>", unsafe_allow_html=True)
