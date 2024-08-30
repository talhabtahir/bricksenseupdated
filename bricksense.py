import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Clear Streamlit cache
st.cache_resource.clear()

# Configure the page
st.set_page_config(
    page_title="Brick Crack Detection",
    page_icon=":brick:",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load the model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('/Kg_33kmodelv36_basev4.keras')
    return model

model = load_model()

# Page Title and Description
st.title("🧱 Brick Crack Detection App")
st.markdown("""
This application uses a deep learning model to detect cracks in brick walls. 
Upload an image of a brick wall, and the model will determine whether the wall is **cracked** or **normal**.
""")

# Sidebar Info
st.sidebar.title("About")
st.sidebar.info(
    """
    This app identifies brick cracks using a pre-trained CNN model. Upload an image to get started.
    """
)

# File Uploader
file = st.file_uploader("Please upload an image of the brick wall (jpg or png):", type=("jpg", "png"))

# Prediction Function
def import_and_predict(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image) / 255.0  # Normalize if required by the model
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

# Handling Image Upload
if file is None:
    st.info("👆 Upload an image file to get started.")
else:
    with st.spinner('Analyzing...'):
        image = Image.open(file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        predictions = import_and_predict(image, model)

        # Debugging outputs (optional, for developer use)
        st.write(f"Predicted raw output: {predictions}")

        probability = predictions[0][0]  # Get the probability value
        # Determine the predicted class based on the probability
        if probability > 0.5:
            predicted_class = "cracked"
            st.error(f"⚠️ This brick wall is likely {predicted_class} with a confidence of {probability:.2f}.")
        else:
            predicted_class = "normal"
            st.success(f"✅ This brick wall appears {predicted_class} with a confidence of {1-probability:.2f}.")
