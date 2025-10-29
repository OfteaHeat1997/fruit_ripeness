"""
Streamlit Web Interface for Fruit Ripeness Classification

This application provides a user-friendly interface for classifying fruit ripeness
using a trained MobileNetV2 deep learning model. Users can upload images or use
their camera to get instant predictions.

Features:
- Camera input for live predictions
- File upload support (jpg, jpeg, png)
- Real-time classification results
- Prediction history logging to SQLite database
- Statistics visualization with charts
"""

import streamlit as st
from PIL import Image
import io
import pandas as pd

# Import custom modules for model predictions and database logging
from src.model_loader import predict_image
from src.db_logging import log_prediction, counts_by_label

# Configure the Streamlit page settings
st.set_page_config(page_title="Fruit Ripeness Classifier", layout="centered")

# Main title of the application
st.title("Fruit Ripeness Classification System")

# Create two tabs: one for predictions, one for statistics
tab1, tab2 = st.tabs(["Predict", "Statistics"])

# ===== TAB 1: Prediction Interface =====
with tab1:
    st.write("Upload an image or use your camera to classify fruit ripeness.")

    # Camera input widget (optional)
    cam_file = st.camera_input("Use Camera (optional)")

    # File uploader widget - accepts jpg, jpeg, png formats
    up_file = st.file_uploader("... or Upload Image", type=["jpg", "jpeg", "png"])

    # Initialize variables for image and filename
    img = None
    filename = None

    # Check if user provided an image via camera
    if cam_file is not None:
        # Open the camera image and convert to RGB format
        img = Image.open(cam_file).convert("RGB")
        filename = "camera.jpg"

    # If no camera image, check if user uploaded a file
    elif up_file is not None:
        # Open the uploaded file and convert to RGB format
        img = Image.open(up_file).convert("RGB")
        filename = up_file.name

    # Process the image if one was provided
    if img is not None:
        # Display the uploaded/captured image
        st.image(img, caption="Received Image", width="stretch")

        # Call the prediction function from src/model_loader.py
        # This function loads the model (if not already loaded) and returns predictions
        out = predict_image(img)

        # Display the prediction results
        st.subheader("Prediction Results")

        # Show the full prediction output as JSON
        # Contains: label (predicted class), score (confidence), all_scores (probabilities for all classes)
        st.json(out)

        # Button to save prediction to database
        if st.button("Save to Database"):
            # Log the prediction with label, confidence score, and metadata
            log_prediction(out["label"], out.get("score"), meta={"filename": filename})

            # Show success message
            st.success("Prediction saved successfully!")

# ===== TAB 2: Statistics Dashboard =====
with tab2:
    st.write("View prediction counts by class from the database.")

    # Retrieve prediction statistics from SQLite database
    rows = counts_by_label()

    # Convert query results to a pandas DataFrame for visualization
    df = pd.DataFrame(rows, columns=["label", "count"])

    # Check if there are any predictions in the database
    if not df.empty:
        # Display bar chart of prediction counts by label
        st.bar_chart(df.set_index("label"))

        # Display the data in a table format
        st.dataframe(df, width="stretch")
    else:
        # Show informational message if no predictions exist yet
        st.info("No predictions yet. Save a prediction in the Predict tab to see statistics.")
