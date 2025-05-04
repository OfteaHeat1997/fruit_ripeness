#!/usr/bin/env python3
"""
Review All Model Predictions - Individual Image Display
Displays each test image with its true label and model prediction
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from PIL import Image

print("="*80)
print("REVIEW ALL MODEL PREDICTIONS")
print("="*80)

# Configuration
MODEL_PATH = 'models/fruit_ripeness_balanced_20251028_215526.keras'
TEST_DIR = 'data/fruit_ripeness_dataset/fruit_ripeness_dataset/fruit_archive/dataset/dataset/test/_clean/_split/test'
IMG_SIZE = (224, 224)
SAMPLES_PER_CLASS = 20  # Change to None to see ALL images

# Class names
CLASS_NAMES = [
    'freshapples',
    'freshbanana',
    'freshoranges',
    'rottenapples',
    'rottenbanana',
    'rottenoranges',
    'unripe apple',
    'unripe banana',
    'unripe orange'
]

print("\n1. Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print(f"   Model loaded successfully!")

print("\n2. Collecting test images...")

def collect_test_images(test_dir, class_names, max_per_class=20):
    """Collect test images from each class"""
    all_images = []

    for idx, class_name in enumerate(class_names):
        class_path = os.path.join(test_dir, class_name)

        # Get all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(Path(class_path).glob(ext))

        # Limit if specified
        if max_per_class and len(image_files) > max_per_class:
            image_files = list(image_files)[:max_per_class]

        for img_path in image_files:
            all_images.append({
                'path': str(img_path),
                'true_label': class_name,
                'true_idx': idx
            })

        print(f"   {class_name}: {len(image_files)} images")

    return all_images

test_images = collect_test_images(TEST_DIR, CLASS_NAMES, SAMPLES_PER_CLASS)
print(f"\n   Total images: {len(test_images)}")

print("\n3. Making predictions...")

def preprocess_image(img_path, img_size=(224, 224)):
    """Load and preprocess image"""
    img = Image.open(img_path).convert('RGB')
    img_resized = img.resize(img_size)
    img_array = np.array(img_resized) / 255.0
    return np.expand_dims(img_array, axis=0), img

predictions_data = []

for i, img_data in enumerate(test_images):
    if (i + 1) % 20 == 0:
        print(f"   Processed {i + 1}/{len(test_images)} images...")

    # Preprocess and predict
    img_batch, original_img = preprocess_image(img_data['path'], IMG_SIZE)
    predictions = model.predict(img_batch, verbose=0)

    # Get results
    pred_idx = np.argmax(predictions[0])
    confidence = predictions[0][pred_idx]
    pred_label = CLASS_NAMES[pred_idx]
    is_correct = (pred_idx == img_data['true_idx'])

    predictions_data.append({
        'path': img_data['path'],
        'image': original_img,
        'true_label': img_data['true_label'],
        'pred_label': pred_label,
        'confidence': confidence,
        'is_correct': is_correct
    })

# Calculate accuracy
correct = sum(1 for p in predictions_data if p['is_correct'])
accuracy = (correct / len(predictions_data)) * 100

print(f"\n   Predictions complete!")
print(f"   Accuracy: {accuracy:.2f}% ({correct}/{len(predictions_data)})")

print("\n4. Displaying all images...")
print("   Green border = CORRECT")
print("   Red border = WRONG")
print("   Close each image window to see the next one")
print("\n" + "="*80 + "\n")

def display_prediction(pred_data, index, total):
    """Display a single image with prediction"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Display image
    ax.imshow(pred_data['image'])
    ax.axis('off')

    # Determine status and color
    if pred_data['is_correct']:
        status = "CORRECT"
        color = 'green'
    else:
        status = "WRONG"
        color = 'red'

    # Add colored border
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(10)

    # Create title
    title = f"Image {index + 1} of {total}\n\n"
    title += f"TRUE LABEL: {pred_data['true_label']}\n"
    title += f"PREDICTED: {pred_data['pred_label']}\n"
    title += f"CONFIDENCE: {pred_data['confidence']:.2%}\n\n"
    title += f"Status: {status}"

    ax.set_title(title, fontsize=16, fontweight='bold', color=color, pad=20)

    plt.tight_layout()

    # Print info to console
    filename = os.path.basename(pred_data['path'])
    print(f"\nImage {index + 1}/{total}")
    print(f"File: {filename}")
    print(f"True: {pred_data['true_label']} | Predicted: {pred_data['pred_label']} | {status}")
    print("-" * 80)

    plt.show()

# Display all predictions
for i, pred_data in enumerate(predictions_data):
    display_prediction(pred_data, i, len(predictions_data))

# Final summary
print("\n\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)
print(f"\nTotal images reviewed: {len(predictions_data)}")
print(f"Correct predictions: {correct}")
print(f"Wrong predictions: {len(predictions_data) - correct}")
print(f"Accuracy: {accuracy:.2f}%")

# Show wrong predictions
wrong_predictions = [p for p in predictions_data if not p['is_correct']]

if wrong_predictions:
    print(f"\n\nINCORRECT PREDICTIONS ({len(wrong_predictions)} total):")
    print("-" * 80)
    for p in wrong_predictions:
        filename = os.path.basename(p['path'])
        print(f"{filename}")
        print(f"  True: {p['true_label']} -> Predicted: {p['pred_label']} ({p['confidence']:.2%})")
else:
    print("\n\nNo incorrect predictions! Perfect accuracy!")

print("\n" + "="*80)
print("REVIEW COMPLETE!")
print("="*80)
