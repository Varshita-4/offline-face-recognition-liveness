
import streamlit as st
import numpy as np
import pickle
import tensorflow as tf

from PIL import Image
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Offline Face Recognition & Liveness Detection",
    layout="wide"
)

st.title(
    " Offline Face Recognition & Liveness Detection"
)

st.write(
    "This demo combines Face Recognition and Face Anti-Spoofing."
)

# Load models
@st.cache_resource
def load_models():

    with open(
        "models/embeddings.pkl",
        "rb"
    ) as f:
        embeddings_database = pickle.load(f)

    liveness_model = load_model(
        "models/liveness_model.h5"
    )

    return embeddings_database, liveness_model


embeddings_database, liveness_model = load_models()

uploaded_file = st.file_uploader(
    "Upload a face image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    img = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        img,
        caption="Uploaded Image",
        use_container_width=True
    )

    img_array = np.array(
        img.resize((224,224))
    )

    img_array = img_array / 255.0
    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    pred = liveness_model.predict(
        img_array,
        verbose=0
    )[0][0]

    st.subheader(
        " Liveness Detection"
    )

    if pred > 0.5:
        st.success(
            f"REAL FACE ({pred*100:.2f}%)"
        )
    else:
        st.error(
            f"FAKE FACE ({(1-pred)*100:.2f}%)"
        )

    st.subheader(
        "Face Recognition"
    )

    st.info(
        "Face Recognition Model (FaceNet) integrated successfully."
    )

    st.subheader(
        " Grad-CAM Visualization"
    )

    try:
        gradcam = Image.open(
            "outputs/gradcam_output.png"
        )

        st.image(
            gradcam,
            caption="Grad-CAM Output",
            use_container_width=True
        )

    except:
        st.warning(
            "Grad-CAM output image not found."
        )
