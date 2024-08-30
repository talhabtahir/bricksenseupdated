from PIL import Image, ImageOps, ExifTags
import numpy as np
import streamlit as st
import tensorflow as tf

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('Kg_33kmodelv36_basev4.keras')
    return model

model = load_model()

st.write("""
# Brick Crack Detection
""")
file = st.file_uploader("Please upload an image of the brick wall", type=("jpg", "png"))

def import_and_predict(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    img = np.asarray(image) / 255.0  # Normalize if required by the model
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

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
        # cases: image don't have getexif
        pass
    return image

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    image = correct_orientation(image)  # Correct the orientation
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)

    # Handle single-value prediction
    probability = predictions[0][0]  # Get the probability value

    # Determine the predicted class based on the probability
    if probability > 0.5:
        predicted_class = "cracked"
    else:
        predicted_class = "normal"
    
    st.write(f"Predicted class: {predicted_class}")

    string = f"This brick wall is {predicted_class}"
    st.success(string)
