"""
Flask REST API for Fruit Ripeness Classification

This API provides endpoints for classifying fruit ripeness using a trained
MobileNetV2 deep learning model. It supports image upload, prediction,
and statistics retrieval.

Endpoints:
- GET  /health         - Health check endpoint
- POST /predict_image  - Upload image and get ripeness prediction
- GET  /stats          - Get prediction statistics from database

The API uses:
- CORS enabled for cross-origin requests
- SQLite database for prediction logging
- GPU acceleration (if available) for fast inference
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io

# Import custom modules for model predictions and database operations
from src.model_loader import predict_image
from src.db_logging import log_prediction, counts_by_label

# Initialize Flask application
app = Flask(__name__)

# Enable CORS (Cross-Origin Resource Sharing) to allow requests from web browsers
CORS(app)


@app.get("/health")
def health():
    """
    Health check endpoint to verify the API is running.

    Returns:
        JSON response with status "ok" and HTTP 200
    """
    return jsonify(status="ok"), 200


@app.post("/predict_image")
def predict_image_endpoint():
    """
    Predict fruit ripeness from an uploaded image.

    Expected request:
        - Method: POST
        - Content-Type: multipart/form-data
        - Field name: "image"
        - Accepted formats: jpg, jpeg, png

    Returns:
        JSON response with prediction results:
        {
            "label": "freshapples",           # Predicted class
            "score": 0.9234,                  # Confidence score (0-1)
            "all_scores": {                   # Probabilities for all classes
                "freshapples": 0.9234,
                "rottenbanana": 0.0123,
                ...
            }
        }

    Error responses:
        - 400: Missing image field
        - 400: Empty file
        - 400: Cannot open image file
    """
    # Check if 'image' field exists in the request
    if "image" not in request.files:
        return jsonify(error="Please upload an image in the 'image' field"), 400

    # Get the uploaded file
    file = request.files["image"]

    # Validate that file is not empty
    if not file or file.filename == "":
        return jsonify(error="Empty file uploaded"), 400

    # Try to open and process the image
    try:
        # Open the image file and convert to RGB format
        # RGB conversion ensures compatibility with the model
        img = Image.open(file.stream).convert("RGB")
    except Exception as e:
        # Return error if image cannot be opened (corrupted, wrong format, etc.)
        return jsonify(error=f"Cannot open image: {e}"), 400

    # Call the prediction function from src/model_loader.py
    # This returns a dictionary with label, score, and all_scores
    out = predict_image(img)

    # Log the prediction to SQLite database for statistics tracking
    # Saves: label, confidence score, and filename as metadata
    log_prediction(out["label"], out.get("score"), meta={"filename": file.filename})

    # Return the prediction results as JSON with HTTP 200 status
    return jsonify(out), 200


@app.get("/stats")
def stats():
    """
    Retrieve prediction statistics from the database.

    Returns:
        JSON response with prediction counts by label:
        {
            "data": {
                "freshapples": 15,
                "rottenbanana": 8,
                "unripe orange": 3
            }
        }

    The counts represent how many times each fruit class has been predicted
    and saved to the database.
    """
    # Query the database for prediction counts grouped by label
    rows = counts_by_label()

    # Convert list of tuples to dictionary format
    # Example: [("freshapples", 15), ("rottenbanana", 8)] -> {"freshapples": 15, "rottenbanana": 8}
    data = {label: cnt for (label, cnt) in rows}

    # Return the statistics as JSON with HTTP 200 status
    return jsonify(data=data), 200


# Run the Flask development server when script is executed directly
if __name__ == "__main__":
    # Configuration:
    # - host="0.0.0.0": Accept connections from any network interface
    # - port=8000: Run on port 8000
    # - debug=True: Enable auto-reload on code changes and detailed error messages
    #
    # WARNING: This is a development server. For production, use a WSGI server
    # like Gunicorn or uWSGI.
    app.run(host="0.0.0.0", port=8000, debug=True)
