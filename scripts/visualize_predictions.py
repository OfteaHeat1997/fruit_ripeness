#!/usr/bin/env python3
"""
Visualize model predictions on test dataset
Creates a grid of images showing true labels, predictions, and confidence scores
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import random
from PIL import Image
from collections import defaultdict

# Configuration
MODEL_PATH = 'models/fruit_ripeness_balanced_20251028_215526.keras'
TEST_DIR = 'data/fruit_ripeness_dataset/fruit_ripeness_dataset/fruit_archive/dataset/dataset/test/_clean/_split/test'
OUTPUT_DIR = 'results/images'
IMG_SIZE = (224, 224)
SAMPLES_PER_CLASS = 5

# Class names (must match training order)
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

print("=" * 80)
print("FRUIT RIPENESS MODEL - PREDICTION VISUALIZATION")
print("=" * 80)

# Load model
print(f"\n1. Loading model from: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)
print(f"   ✓ Model loaded successfully")
print(f"   Input shape: {model.input_shape}")
print(f"   Output shape: {model.output_shape}")

# Load sample images
print(f"\n2. Loading sample images from test dataset...")
print(f"   Test directory: {TEST_DIR}")

def load_sample_images(test_dir, class_names, samples_per_class=5):
    """Load sample images from each class"""
    samples = []

    for idx, class_name in enumerate(class_names):
        class_path = os.path.join(test_dir, class_name)

        # Get all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(Path(class_path).glob(ext))

        # Sample images
        if len(image_files) > 0:
            num_samples = min(samples_per_class, len(image_files))
            sampled_files = random.sample(image_files, num_samples)

            for img_path in sampled_files:
                samples.append((str(img_path), class_name, idx))

            print(f"   {class_name}: {len(image_files)} images, sampled {num_samples}")
        else:
            print(f"   {class_name}: No images found")

    return samples

sample_images = load_sample_images(TEST_DIR, CLASS_NAMES, SAMPLES_PER_CLASS)
print(f"\n   Total samples: {len(sample_images)}")

# Make predictions
print(f"\n3. Making predictions on {len(sample_images)} images...")

def preprocess_image(img_path, img_size=(224, 224)):
    """Load and preprocess image"""
    img = Image.open(img_path).convert('RGB')
    img = img.resize(img_size)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0), img

predictions_data = []

for img_path, true_label, true_idx in sample_images:
    # Preprocess and predict
    img_batch, original_img = preprocess_image(img_path, IMG_SIZE)
    predictions = model.predict(img_batch, verbose=0)

    # Get results
    pred_idx = np.argmax(predictions[0])
    confidence = predictions[0][pred_idx]
    pred_label = CLASS_NAMES[pred_idx]
    is_correct = (pred_idx == true_idx)

    predictions_data.append({
        'img_path': img_path,
        'img': original_img,
        'true_label': true_label,
        'true_idx': true_idx,
        'pred_label': pred_label,
        'pred_idx': pred_idx,
        'confidence': confidence,
        'is_correct': is_correct
    })

print(f"   ✓ Predictions completed")

# Calculate accuracy
correct = sum(1 for p in predictions_data if p['is_correct'])
accuracy = (correct / len(predictions_data)) * 100
print(f"\n   Accuracy: {accuracy:.2f}% ({correct}/{len(predictions_data)})")

# Create visualization
print(f"\n4. Creating visualization grid...")

def plot_predictions_grid(predictions_data, cols=5, figsize=(20, 36)):
    """Plot grid of images with predictions"""
    num_images = len(predictions_data)
    rows = (num_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if num_images > 1 else [axes]

    for idx, pred_data in enumerate(predictions_data):
        ax = axes[idx]

        # Display image
        ax.imshow(pred_data['img'])

        # Set border color
        border_color = 'green' if pred_data['is_correct'] else 'red'
        border_width = 5

        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(border_width)

        # Create title
        title = f"True: {pred_data['true_label']}\n"
        title += f"Pred: {pred_data['pred_label']}\n"
        title += f"Conf: {pred_data['confidence']:.2%}"

        ax.set_title(title, fontsize=10, fontweight='bold', color=border_color)
        ax.axis('off')

    # Hide empty subplots
    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    # Save figure
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, 'model_predictions_grid.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Grid saved to: {output_path}")

    # Try to show if in interactive mode
    try:
        plt.show()
    except:
        pass

plot_predictions_grid(predictions_data, cols=5)

# Print class-by-class statistics
print(f"\n5. Class-by-class performance:")
print("=" * 80)

class_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'confidences': []})

for pred_data in predictions_data:
    true_label = pred_data['true_label']
    class_stats[true_label]['total'] += 1
    class_stats[true_label]['confidences'].append(pred_data['confidence'])

    if pred_data['is_correct']:
        class_stats[true_label]['correct'] += 1

for class_name in CLASS_NAMES:
    if class_name in class_stats:
        stats = class_stats[class_name]
        acc = (stats['correct'] / stats['total']) * 100
        avg_conf = np.mean(stats['confidences']) * 100

        print(f"\n{class_name}:")
        print(f"  Accuracy: {acc:.1f}% ({stats['correct']}/{stats['total']})")
        print(f"  Avg Confidence: {avg_conf:.1f}%")

# Analyze misclassifications
print(f"\n6. Misclassification analysis:")
print("=" * 80)

misclassified = [p for p in predictions_data if not p['is_correct']]

if len(misclassified) > 0:
    print(f"\nTotal misclassifications: {len(misclassified)}")
    print("\nConfusion patterns:")

    confusion_pairs = defaultdict(int)
    for p in misclassified:
        pair = (p['true_label'], p['pred_label'])
        confusion_pairs[pair] += 1

    sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)

    for (true_label, pred_label), count in sorted_pairs:
        print(f"  {true_label} → {pred_label}: {count} times")
else:
    print("\nNo misclassifications! Perfect accuracy!")

# Overall summary
print(f"\n7. Overall summary:")
print("=" * 80)
print(f"\nModel: {MODEL_PATH}")
print(f"Test samples: {len(predictions_data)}")
print(f"Correct: {correct}")
print(f"Incorrect: {len(predictions_data) - correct}")
print(f"Overall accuracy: {accuracy:.2f}%")

avg_confidence = np.mean([p['confidence'] for p in predictions_data]) * 100
print(f"Average confidence: {avg_confidence:.2f}%")

correct_confidences = [p['confidence'] for p in predictions_data if p['is_correct']]
incorrect_confidences = [p['confidence'] for p in predictions_data if not p['is_correct']]

if correct_confidences:
    print(f"Avg confidence (correct): {np.mean(correct_confidences)*100:.2f}%")
if incorrect_confidences:
    print(f"Avg confidence (incorrect): {np.mean(incorrect_confidences)*100:.2f}%")

print("\n" + "=" * 80)
print("VISUALIZATION COMPLETE!")
print("=" * 80)
print(f"\nOutput saved to: {OUTPUT_DIR}/model_predictions_grid.png")
print("Green border = Correct prediction")
print("Red border = Incorrect prediction")
