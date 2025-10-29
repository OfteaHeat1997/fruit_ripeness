"""
Dataset Analysis Script - Predict on All Images

This script runs predictions on all images in the dataset and displays:
- Each image with prediction results
- Confidence percentages for all classes
- Comparison with ground truth labels
- Overall accuracy metrics

Usage:
    python analyze_dataset.py
"""

import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd

# Import our prediction function
from src.model_loader import predict_image

# ===== Configuration =====

# Dataset path - adjust if needed
DATASET_PATH = "data/fruit_ripeness_dataset/fruit_ripeness_dataset/fruit_archive/dataset/dataset/test/_clean/_split/test"

# Number of images to display at once (set to None to show all)
MAX_DISPLAY = None

# Figure size for each image
FIG_SIZE = (15, 8)

# ===== Helper Functions =====

def get_ground_truth_from_path(image_path):
    """
    Extract the ground truth label from the image file path.

    The dataset structure is: .../test/class_name/image.jpg
    So the parent directory name is the ground truth label.

    Args:
        image_path (str): Full path to the image

    Returns:
        str: Ground truth label (parent directory name)
    """
    return os.path.basename(os.path.dirname(image_path))


def plot_prediction_results(img, result, ground_truth, image_name):
    """
    Display an image with its prediction results.

    Creates a visualization with:
    - Left: Original image
    - Right: Bar chart of all class probabilities

    Args:
        img (PIL.Image): The image
        result (dict): Prediction results from predict_image()
        ground_truth (str): Actual class label from folder structure
        image_name (str): Name of the image file
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_SIZE)

    # Left: Display image
    ax1.imshow(img)
    ax1.axis('off')

    # Determine if prediction is correct
    is_correct = result['label'] == ground_truth
    color = 'green' if is_correct else 'red'
    status = 'CORRECT' if is_correct else 'INCORRECT'

    # Title with prediction info
    ax1.set_title(
        f"Image: {image_name}\n"
        f"Ground Truth: {ground_truth}\n"
        f"Predicted: {result['label']} ({result['score']*100:.1f}%)\n"
        f"{status}",
        fontsize=12,
        fontweight='bold',
        color=color
    )

    # Right: Bar chart of probabilities
    scores_df = pd.DataFrame([
        {'Class': k, 'Probability': v}
        for k, v in result['all_scores'].items()
    ]).sort_values('Probability', ascending=True)

    # Color bars: green for predicted, blue for ground truth, gray for others
    colors = []
    for cls in scores_df['Class']:
        if cls == result['label']:
            colors.append('green')
        elif cls == ground_truth:
            colors.append('orange')
        else:
            colors.append('lightgray')

    bars = ax2.barh(scores_df['Class'], scores_df['Probability'], color=colors)
    ax2.set_xlabel('Probability', fontsize=11)
    ax2.set_ylabel('Class', fontsize=11)
    ax2.set_title('Prediction Probabilities', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.grid(axis='x', alpha=0.3)

    # Add percentage labels on bars
    for i, (bar, prob) in enumerate(zip(bars, scores_df['Probability'])):
        if prob > 0.05:  # Only show label if probability > 5%
            ax2.text(prob, bar.get_y() + bar.get_height()/2,
                    f'{prob*100:.1f}%',
                    ha='left', va='center', fontsize=9)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Predicted'),
        Patch(facecolor='orange', label='Ground Truth'),
        Patch(facecolor='lightgray', label='Other Classes')
    ]
    ax2.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.show()


def analyze_dataset(dataset_path, max_images=None, display_images=True):
    """
    Analyze all images in the dataset with predictions.

    Args:
        dataset_path (str): Path to the test dataset
        max_images (int, optional): Maximum number of images to process
        display_images (bool): Whether to display each image with predictions

    Returns:
        pd.DataFrame: Results dataframe with all predictions and metrics
    """
    print(f"Searching for images in: {dataset_path}")

    # Find all images in the dataset
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    all_images = []

    for ext in image_extensions:
        all_images.extend(glob.glob(os.path.join(dataset_path, '**', ext), recursive=True))

    if not all_images:
        print(f"No images found in {dataset_path}")
        return None

    print(f"Found {len(all_images)} images")

    # Limit number of images if specified
    if max_images:
        all_images = all_images[:max_images]
        print(f"Limited to first {max_images} images for analysis")

    # Process each image
    results = []
    print("\nRunning predictions...")

    for img_path in tqdm(all_images, desc="Processing images"):
        try:
            # Load image
            img = Image.open(img_path).convert('RGB')

            # Get ground truth from folder name
            ground_truth = get_ground_truth_from_path(img_path)

            # Run prediction
            prediction = predict_image(img)

            # Check if prediction is correct
            is_correct = prediction['label'] == ground_truth

            # Store results
            results.append({
                'image_path': img_path,
                'image_name': os.path.basename(img_path),
                'ground_truth': ground_truth,
                'predicted_label': prediction['label'],
                'confidence': prediction['score'],
                'correct': is_correct,
                'all_scores': prediction['all_scores']
            })

            # Display image with predictions if requested
            if display_images:
                plot_prediction_results(img, prediction, ground_truth, os.path.basename(img_path))

        except Exception as e:
            print(f"\nError processing {img_path}: {e}")
            continue

    # Convert results to DataFrame
    df = pd.DataFrame(results)

    return df


def print_summary_statistics(df):
    """
    Print summary statistics of the prediction results.

    Args:
        df (pd.DataFrame): Results dataframe
    """
    print("\n" + "="*70)
    print("DATASET ANALYSIS SUMMARY")
    print("="*70)

    # Overall accuracy
    accuracy = df['correct'].mean() * 100
    print(f"\nOverall Accuracy: {accuracy:.2f}%")
    print(f"Correct Predictions: {df['correct'].sum()}/{len(df)}")
    print(f"Incorrect Predictions: {(~df['correct']).sum()}/{len(df)}")

    # Average confidence
    avg_confidence = df['confidence'].mean() * 100
    print(f"\nAverage Confidence: {avg_confidence:.2f}%")
    print(f"   Min Confidence: {df['confidence'].min()*100:.2f}%")
    print(f"   Max Confidence: {df['confidence'].max()*100:.2f}%")

    # Accuracy by class
    print("\nAccuracy by Ground Truth Class:")
    class_accuracy = df.groupby('ground_truth').agg({
        'correct': ['sum', 'count', 'mean']
    }).round(4)
    class_accuracy.columns = ['Correct', 'Total', 'Accuracy']
    class_accuracy['Accuracy'] = class_accuracy['Accuracy'] * 100
    class_accuracy = class_accuracy.sort_values('Accuracy', ascending=False)
    print(class_accuracy.to_string())

    # Confusion patterns (most common mistakes)
    print("\nMost Common Misclassifications:")
    incorrect = df[~df['correct']]
    if len(incorrect) > 0:
        mistakes = incorrect.groupby(['ground_truth', 'predicted_label']).size()
        mistakes = mistakes.sort_values(ascending=False).head(10)
        print("\n   Ground Truth → Predicted (Count)")
        for (gt, pred), count in mistakes.items():
            print(f"   {gt:20s} → {pred:20s} ({count} times)")
    else:
        print("   🎉 No misclassifications! Perfect accuracy!")

    # Confidence distribution for correct vs incorrect
    print("\nConfidence Distribution:")
    print(f"   Correct predictions:   {df[df['correct']]['confidence'].mean()*100:.2f}% avg confidence")
    print(f"   Incorrect predictions: {df[~df['correct']]['confidence'].mean()*100:.2f}% avg confidence")

    print("\n" + "="*70)


def create_confusion_matrix(df):
    """
    Create and display a confusion matrix visualization.

    Args:
        df (pd.DataFrame): Results dataframe
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    # Get unique classes
    classes = sorted(df['ground_truth'].unique())

    # Create confusion matrix
    cm = confusion_matrix(df['ground_truth'], df['predicted_label'], labels=classes)

    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('Ground Truth', fontsize=12)
    plt.title('Confusion Matrix - Dataset Analysis', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_confidence_distribution(df):
    """
    Plot the distribution of prediction confidences.

    Args:
        df (pd.DataFrame): Results dataframe
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Overall confidence distribution
    axes[0].hist(df['confidence'], bins=20, edgecolor='black', alpha=0.7)
    axes[0].axvline(df['confidence'].mean(), color='red', linestyle='--',
                    label=f'Mean: {df["confidence"].mean():.3f}')
    axes[0].set_xlabel('Confidence Score', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Overall Confidence Distribution', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Right: Confidence for correct vs incorrect
    correct_conf = df[df['correct']]['confidence']
    incorrect_conf = df[~df['correct']]['confidence']

    axes[1].hist(correct_conf, bins=15, alpha=0.7, label='Correct', color='green', edgecolor='black')
    axes[1].hist(incorrect_conf, bins=15, alpha=0.7, label='Incorrect', color='red', edgecolor='black')
    axes[1].set_xlabel('Confidence Score', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title('Confidence: Correct vs Incorrect', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# ===== Main Execution =====

if __name__ == "__main__":
    print("="*70)
    print("FRUIT RIPENESS DATASET ANALYSIS")
    print("="*70)

    # Check if dataset path exists
    if not os.path.exists(DATASET_PATH):
        print(f"\nDataset path not found: {DATASET_PATH}")
        print("\nPlease update DATASET_PATH in the script to point to your test dataset.")
        exit(1)

    # Ask user for preferences
    print("\nConfiguration:")
    print(f"   Dataset path: {DATASET_PATH}")
    print(f"   Max images to display: {MAX_DISPLAY if MAX_DISPLAY else 'All'}")

    response = input("\nDo you want to display each image with predictions? (y/n) [y]: ").lower()
    display_images = response != 'n'

    if MAX_DISPLAY is None:
        response = input("How many images to analyze? (Enter number or 'all') [all]: ").lower()
        if response and response != 'all':
            try:
                MAX_DISPLAY = int(response)
            except:
                print("Invalid number, using all images")

    # Run analysis
    results_df = analyze_dataset(
        DATASET_PATH,
        max_images=MAX_DISPLAY,
        display_images=display_images
    )

    if results_df is not None and len(results_df) > 0:
        # Print summary statistics
        print_summary_statistics(results_df)

        # Create visualizations
        print("\nGenerating visualizations...")

        # Confusion matrix
        create_confusion_matrix(results_df)

        # Confidence distribution
        plot_confidence_distribution(results_df)

        # Save results to CSV
        output_file = "dataset_analysis_results.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")

        print("\nAnalysis complete!")
    else:
        print("\nNo results to analyze")
