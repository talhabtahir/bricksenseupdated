import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, ExifTags
import numpy as np

# Set the page configuration
st.set_page_config(page_title="Brick Crack Detection", page_icon=":brick:", layout="centered")

# Custom CSS for additional styling
st.markdown(
    """
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
    model = tf.keras.models.load_model('Kg_33kmodelv36_basev4.keras')
    return model

model = load_model()

# Sidebar for user input and information
st.sidebar.header("Upload Image")
file = st.sidebar.file_uploader("Please upload an image of the brick wall", type=("jpg", "png"))

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

# Function to make predictions using the model
def import_and_predict(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    img = np.asarray(image) / 255.0  # Normalize if required by the model
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.info("Please upload an image file to start the detection.")
else:
    image = Image.open(file)
    image = correct_orientation(image)  # Correct the orientation

    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Prediction and display of results
    st.write("Analyzing the image...")
    predictions = import_and_predict(image, model)
    probability = predictions[0][0]

    if probability > 0.5:
        predicted_class = "cracked"
        st.error(f"⚠️ This brick wall is {predicted_class}.")
    else:
        predicted_class = "normal"
        st.success(f"✅ This brick wall is {predicted_class}.")

    # Display predicted probability
    st.write(f"**Predicted Probability of being cracked:** {probability:.2f}")

# Footer
st.markdown("<div class='footer'>Developed with Streamlit & TensorFlow | © 2024 BrickSense</div>", unsafe_allow_html=True)
