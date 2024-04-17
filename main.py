import streamlit as st
from PIL import Image
import io
import numpy as np
import tensorflow as tf
from utils import clean_image, get_prediction, make_results

# Loading the Model and saving to cache
@st.cache_resource
def load_model(path):
    # Xception Model
    xception_model = tf.keras.models.Sequential([
        tf.keras.applications.xception.Xception(include_top=False, weights='imagenet', input_shape=(512, 512, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(4, activation='softmax')
    ])

    # DenseNet Model
    densenet_model = tf.keras.models.Sequential([
        tf.keras.applications.densenet.DenseNet121(include_top=False, weights='imagenet', input_shape=(512, 512, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(4, activation='softmax')
    ])

    # Ensembling the Models
    inputs = tf.keras.Input(shape=(512, 512, 3))
    xception_output = xception_model(inputs)
    densenet_output = densenet_model(inputs)
    outputs = tf.keras.layers.average([densenet_output, xception_output])
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Loading the Weights of Model
    model.load_weights(path)
    
    return model

# Remove Menu
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

# Page title and sidebar
st.title("Potato Leaf Disease Detection")
st.sidebar.title("Navigation")

# Sidebar navigation
page_selection = st.sidebar.radio(
    "Go to",
    ("Home", "Upload Image", "About")
)

# Home page content
if page_selection == "Home":
    st.write(
        """
        <style>
        .main-container {
            background: linear-gradient(to right, #ff5252, #ffcc80);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .title-text {
            color: #ffffff;
        }
        .body-text {
            font-size: 18px;
            color: #ffffff;
        }
        </style>
        <div class="main-container">
        <h2 class="title-text">Welcome to the Potato Leaf Disease Detector!</h2>
        <p class="body-text">Use the sidebar to navigate or upload an image to get started.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# About page content
elif page_selection == "About":
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)
    st.markdown("<h2 class='title-text'>About Potato Leaf Disease Detection System</h2>", unsafe_allow_html=True)
    st.markdown("<p class='body-text'>Potato Leaf Disease Detection System is a machine learning application that helps in identifying common diseases affecting potato plants based on images of their leaves.</p>", unsafe_allow_html=True)
    st.markdown("<h3 class='title-text'>How it Works</h3>", unsafe_allow_html=True)
    st.markdown("<ul class='body-text'>", unsafe_allow_html=True)
    st.markdown("<li>Collect a large dataset of potato leaf images covering various conditions, including healthy leaves and leaves affected by diseases.</li>", unsafe_allow_html=True)
    st.markdown("<li>Preprocess the dataset by resizing, normalization, and augmentation.</li>", unsafe_allow_html=True)
    st.markdown("<li>Build a deep learning model using convolutional neural networks (CNNs) to classify diseases.</li>", unsafe_allow_html=True)
    st.markdown("<li>Train the model on the preprocessed dataset and evaluate its performance.</li>", unsafe_allow_html=True)
    st.markdown("<li>Deploy the trained model as a web application using Streamlit.</li>", unsafe_allow_html=True)
    st.markdown("</ul>", unsafe_allow_html=True)
    st.markdown("<h3 class='title-text'>Benefits</h3>", unsafe_allow_html=True)
    st.markdown("<ul class='body-text'>", unsafe_allow_html=True)
    st.markdown("<li>Early detection of diseases helps farmers take timely preventive measures to protect their crops.</li>", unsafe_allow_html=True)
    st.markdown("<li>Reduces yield losses and optimizes resource utilization.</li>", unsafe_allow_html=True)
    st.markdown("</ul>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# Upload image page content
elif page_selection == "Upload Image":
    st.subheader("Upload Image")
    uploaded_image = st.file_uploader(
        "Upload an image of a potato leaf",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is not None:
        try:
            image = Image.open(io.BytesIO(uploaded_image.read()))
            st.image(image, caption="Uploaded Potato Leaf Image", use_column_width=True)
            
            # Display progress and text
            progress = st.text("Crunching Image")
            my_bar = st.progress(0)

            # Cleaning the image
            image = clean_image(image)

            # Making the predictions
            model = load_model('model.h5')
            predictions, predictions_arr = get_prediction(model, image)
            my_bar.progress(70)

            # Making the results
            result = make_results(predictions, predictions_arr)

            # Removing progress bar and text after prediction is done
            my_bar.progress(100)
            progress.empty()
            my_bar.empty()

            # Show the results
            st.write(f"The plant is {result['status']} with {result['prediction']} prediction.")
        except Exception as e:
            st.error("Error: Unable to process the image. Please make sure it's a valid image file.")
