import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, ExifTags
import numpy as np
import torch
import cv2
import ultralytics

# Set the page configuration with favicon
st.set_page_config(
    page_title="Brick Crack Detection",
    page_icon="static/brickicon4.png",  # Path to your favicon file
    layout="centered"
)

# Custom CSS for additional styling
st.markdown(
    """
    <link rel="icon" href="static/brickicon4.png" type="image/x-icon">
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

# Header with an icon
st.markdown("<h1 class='main-header'>🧱 Brick Crack Detection 🧱</h1>", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('Kg_33kmodelv36_basev4.keras')

@st.cache_resource
def load_yolo_model():
    return torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

model = load_model()
yolo_model = load_yolo_model()

# Sidebar for app information
st.sidebar.header("About This App")
st.sidebar.write("""
This app uses a Convolutional Neural Network (CNN) model to detect cracks in brick walls. 
You can upload an image of a brick wall, and the app will analyze the image to determine if the wall is cracked or normal. 
The output will provide a predicted class and the probability of the wall being cracked or normal.
""")
st.sidebar.write("""
**Developed by:**  
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

# Function to make predictions using the TensorFlow model
def import_and_predict(image_data, model):
    try:
        size = (224, 224)
        # Convert image to RGB
        image = image_data.convert("RGB")
        image = ImageOps.fit(image, size, Image.LANCZOS)
        img = np.asarray(image).astype(np.float32) / 255.0
        img_reshape = img[np.newaxis, ...]  # Add batch dimension
        prediction = model.predict(img_reshape)
        return prediction
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return None

# Function to analyze image with YOLOv5
def analyze_with_yolo(image_path):
    try:
        img_cv2 = cv2.imread(image_path)
        if img_cv2 is None:
            st.error(f"Error: Could not open or find the image at {image_path}")
            return None
        img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        img_rgb = np.ascontiguousarray(img_rgb)
        results = yolo_model(img_rgb)
        results_df = results.pandas().xyxy[0]
        return results_df
    except Exception as e:
        st.error(f"An error occurred during YOLO analysis: {e}")
        return None

if file is None:
    st.info("Please upload an image file to start the detection.")
else:
    image = Image.open(file)
    image = correct_orientation(image)  # Correct the orientation

    # Save the uploaded image temporarily for YOLO processing
    image_path = '/tmp/uploaded_image.jpg'
    image.save(image_path)

    # Analyze with YOLOv5
    yolo_results = analyze_with_yolo(image_path)
    
    if yolo_results is not None and not yolo_results.empty:
        high_confidence_results = yolo_results[yolo_results['confidence'] > 0.8]
        if not high_confidence_results.empty:
            detected_classes = high_confidence_results['name'].unique()
            detected_classes_str = ', '.join(detected_classes)
            st.write(f"YOLOv5 detected the following classes with high confidence: {detected_classes_str}")
            st.warning(f"{detected_classes_str} detected in the uploaded picture. Please upload an image of a brick wall.")
        else:
            # Proceed with TensorFlow model prediction
            predictions = import_and_predict(image, model)
            if predictions is not None:
                probability = predictions[0][0]
                if probability > 0.5:
                    predicted_class = "cracked"
                    st.error(f"⚠️ This brick wall is {predicted_class}.")
                    st.write(f"**Predicted Probability:** {probability * 100:.2f}% cracked.")
                else:
                    predicted_class = "normal"
                    st.success(f"✅ This brick wall is {predicted_class}.")
                    st.write(f"**Predicted Probability:** {(1 - probability) * 100:.2f}% normal.")
    else:
        st.error("Error processing image with YOLOv5.")

# Footer
st.markdown("<div class='footer'>Developed with Streamlit & TensorFlow | © 2024 BrickSense</div>", unsafe_allow_html=True)
