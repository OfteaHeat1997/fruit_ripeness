"""
Retrain Fruit Ripeness Model with Class Balancing

This script automatically trains a new model using your existing dataset.
NO manual image upload needed - uses images already on disk!

Features:
- Automatic class weight balancing
- Data augmentation (rotation, zoom, flip)
- GPU acceleration (if available)
- Progress tracking
- Automatic model saving
- Early stopping (stops when no improvement)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow verbosity

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    CSVLogger,
    TensorBoard
)
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from pathlib import Path
import json
from datetime import datetime

print("=" * 70)
print("FRUIT RIPENESS MODEL - BALANCED TRAINING")
print("=" * 70)
print()

# ============================================================================
# PART 1: CONFIGURATION
# ============================================================================
print("PART 1: Configuration")
print("-" * 70)

# Paths (no need to change these - automatic!)
# FULL DATASET with ALL 9 categories (including unripe fruits!)
TRAIN_DIR = "data/fruit_ripeness_dataset/fruit_ripeness_dataset/fruit_archive/dataset/train"
TEST_DIR = "data/fruit_ripeness_dataset/fruit_ripeness_dataset/fruit_archive/dataset/test"

# Training parameters
IMG_SIZE = 224        # Image size (224x224 pixels)
BATCH_SIZE = 32       # How many images to process at once
EPOCHS = 50           # Maximum training rounds
LEARNING_RATE = 0.0001  # How fast the model learns

# Model output
OUTPUT_DIR = "models"
MODEL_NAME = f"fruit_ripeness_balanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras"

print(f"Training data: {TRAIN_DIR}")
print(f"Test data: {TEST_DIR}")
print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Max epochs: {EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Output: {OUTPUT_DIR}/{MODEL_NAME}")
print()

# Check GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU Available: {gpus[0].name}")
    print(f"   Using GPU acceleration!")
else:
    print("No GPU detected - using CPU (slower)")
print()

# ============================================================================
# PART 2: DATA LOADING & AUGMENTATION
# ============================================================================
print("PART 2: Data Loading & Augmentation")
print("-" * 70)

# Data augmentation for training (creates variations of images)
# This helps the model learn better by seeing images from different angles
print("Setting up data augmentation...")
print("   - Random rotation: ±20 degrees")
print("   - Random shift: 20% horizontal/vertical")
print("   - Random zoom: ±20%")
print("   - Random horizontal flip")
print("   - Brightness adjustment: ±20%")
print()

train_datagen = ImageDataGenerator(
    rescale=1./255,                    # Normalize pixel values to 0-1
    rotation_range=20,                 # Randomly rotate images
    width_shift_range=0.2,             # Randomly shift images horizontally
    height_shift_range=0.2,            # Randomly shift images vertically
    shear_range=0.2,                   # Randomly shear images
    zoom_range=0.2,                    # Randomly zoom in/out
    horizontal_flip=True,              # Randomly flip images
    brightness_range=[0.8, 1.2],       # Randomly adjust brightness
    fill_mode='nearest'                # Fill empty pixels after transformation
)

# No augmentation for test data (we want original images)
test_datagen = ImageDataGenerator(
    rescale=1./255
)

# Load training data
print("📥 Loading training data...")
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

print(f"Found {train_generator.samples} training images")
print(f"Found {len(train_generator.class_indices)} classes:")
for class_name, class_idx in sorted(train_generator.class_indices.items(), key=lambda x: x[1]):
    count = np.sum(train_generator.classes == class_idx)
    pct = (count / train_generator.samples) * 100
    print(f"   {class_idx}. {class_name:20s}: {count:4d} images ({pct:5.1f}%)")
print()

# Load test/validation data
print("📥 Loading test data...")
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)
print(f"Found {test_generator.samples} test images")
print()

num_classes = len(train_generator.class_indices)

# ============================================================================
# PART 3: CLASS WEIGHT CALCULATION (BALANCING)
# ============================================================================
print("PART 3: Class Weight Calculation")
print("-" * 70)
print("Calculating class weights to balance training...")
print()
print("💡 What are class weights?")
print("   Class weights tell the model to pay MORE attention to rare classes")
print("   and LESS attention to common classes. This prevents the model from")
print("   only learning the majority class.")
print()

# Calculate class weights (automatic balancing!)
unique_classes = np.unique(train_generator.classes)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=unique_classes,
    y=train_generator.classes
)
# Map weights to actual class indices (not just 0, 1, 2...)
class_weight_dict = {cls: weight for cls, weight in zip(unique_classes, class_weights)}

print("Calculated Class Weights:")
print("   (Higher weight = model pays more attention)")
print()
for class_name, class_idx in sorted(train_generator.class_indices.items(), key=lambda x: x[1]):
    # Only print classes that actually have samples
    if class_idx in class_weight_dict:
        weight = class_weight_dict[class_idx]
        print(f"   {class_name:20s}: {weight:.3f}")
    else:
        print(f"   {class_name:20s}: SKIPPED (no samples)")
print()

# ============================================================================
# PART 4: MODEL ARCHITECTURE
# ============================================================================
print("PART 4: Model Architecture")
print("-" * 70)
print("🏗️  Building MobileNetV2 model...")
print()

# Load pre-trained MobileNetV2 (trained on ImageNet)
# This is called "transfer learning" - we use knowledge from another task
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,  # Remove the classification layer
    weights='imagenet'  # Use ImageNet pre-trained weights
)

# Freeze the base model (don't train it initially)
base_model.trainable = False

print("Loaded MobileNetV2 base model")
print(f"   Parameters: {base_model.count_params():,}")
print()

# Add our custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)              # Reduce dimensions
x = Dense(256, activation='relu')(x)         # Hidden layer
x = Dropout(0.5)(x)                          # Prevent overfitting
outputs = Dense(num_classes, activation='softmax')(x)  # Output layer

model = Model(inputs=base_model.input, outputs=outputs)

print("Added custom classification layers")
print(f"   Total parameters: {model.count_params():,}")
print()

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Model compiled and ready to train!")
print()

# ============================================================================
# PART 5: TRAINING CALLBACKS
# ============================================================================
print("PART 5: Training Callbacks (Automatic Helpers)")
print("-" * 70)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create TensorBoard log directory
tensorboard_log_dir = os.path.join("logs", "fit", datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(tensorboard_log_dir, exist_ok=True)

# Setup callbacks
callbacks = [
    # Save best model automatically
    ModelCheckpoint(
        filepath=os.path.join(OUTPUT_DIR, MODEL_NAME),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),

    # Stop training if no improvement
    EarlyStopping(
        monitor='val_accuracy',
        patience=10,  # Stop after 10 epochs with no improvement
        restore_best_weights=True,
        verbose=1
    ),

    # Reduce learning rate if stuck
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,  # Reduce by half
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),

    # Log training history to CSV
    CSVLogger(
        os.path.join(OUTPUT_DIR, f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    ),

    # TensorBoard for live visualization
    TensorBoard(
        log_dir=tensorboard_log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=False,
        update_freq='epoch',
        profile_batch=0
    )
]

print("Callbacks configured:")
print("   - ModelCheckpoint: Saves best model automatically")
print("   - EarlyStopping: Stops if no improvement (patience=10)")
print("   - ReduceLROnPlateau: Reduces learning rate if stuck")
print("   - CSVLogger: Saves training history")
print("   - TensorBoard: Live training visualization")
print()
print(f"TensorBoard Logs: {tensorboard_log_dir}")
print("   To view live training dashboard, run in another terminal:")
print(f"   tensorboard --logdir={tensorboard_log_dir}")
print()

# ============================================================================
# PART 6: TRAINING PHASE 1 (Top Layers Only)
# ============================================================================
print("PART 6: Training Phase 1 - Top Layers Only")
print("-" * 70)
print("🏃 Starting training (this will take a while)...")
print()
print("💡 Phase 1 Strategy:")
print("   We train only the NEW layers we added (classification head)")
print("   The base MobileNetV2 stays frozen (using pre-trained weights)")
print("   This is faster and prevents overfitting.")
print()

history1 = model.fit(
    train_generator,
    epochs=10,  # Train for 10 epochs
    validation_data=test_generator,
    class_weight=class_weight_dict,  # Apply class weights!
    callbacks=callbacks,
    verbose=1
)

print()
print("Phase 1 complete!")
print(f"   Best accuracy: {max(history1.history['val_accuracy'])*100:.2f}%")
print()

# ============================================================================
# PART 7: TRAINING PHASE 2 (Fine-tuning)
# ============================================================================
print("PART 7: Training Phase 2 - Fine-Tuning")
print("-" * 70)
print("🔓 Unfreezing base model for fine-tuning...")
print()
print("💡 Phase 2 Strategy:")
print("   Now we unfreeze the base model and train the ENTIRE network")
print("   We use a lower learning rate to avoid destroying the pre-trained weights")
print()

# Unfreeze the base model
base_model.trainable = True

# Recompile with lower learning rate
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE / 10),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Model unfrozen and recompiled with lower learning rate")
print()

# Continue training
print("🏃 Continuing training with full model...")
print()

history2 = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    class_weight=class_weight_dict,  # Still using class weights
    callbacks=callbacks,
    verbose=1,
    initial_epoch=len(history1.history['loss'])  # Continue from where we left off
)

print()
print("Phase 2 complete!")
print()

# ============================================================================
# PART 8: EVALUATION
# ============================================================================
print("=" * 70)
print("FINAL EVALUATION")
print("=" * 70)

# Evaluate on test set
test_loss, test_acc = model.evaluate(test_generator, verbose=0)

print(f"\nTest Results:")
print(f"   Accuracy: {test_acc*100:.2f}%")
print(f"   Loss: {test_loss:.4f}")
print()

# Compare with old model (if you know the old accuracy)
old_accuracy = 21.87  # From your analysis
improvement = test_acc * 100 - old_accuracy

print(f"Improvement:")
print(f"   Old model: {old_accuracy:.2f}%")
print(f"   New model: {test_acc*100:.2f}%")
if improvement > 0:
    print(f"   🎉 Improvement: +{improvement:.2f}% !")
else:
    print(f"   Change: {improvement:.2f}%")
print()

# Save final model
final_model_path = os.path.join(OUTPUT_DIR, "fruit_ripeness_final.keras")
model.save(final_model_path)

print(f"Model saved:")
print(f"   Best: {OUTPUT_DIR}/{MODEL_NAME}")
print(f"   Final: {final_model_path}")
print()

# Save class labels
class_labels = {v: k for k, v in train_generator.class_indices.items()}
labels_path = os.path.join(OUTPUT_DIR, "class_labels.json")
with open(labels_path, 'w') as f:
    json.dump(class_labels, f, indent=2)
print(f"   Labels: {labels_path}")
print()

# Save training info
training_info = {
    "training_date": datetime.now().isoformat(),
    "num_classes": num_classes,
    "class_names": list(train_generator.class_indices.keys()),
    "num_training_images": train_generator.samples,
    "num_test_images": test_generator.samples,
    "image_size": IMG_SIZE,
    "batch_size": BATCH_SIZE,
    "epochs_trained": len(history1.history['loss']) + len(history2.history['loss']),
    "final_test_accuracy": float(test_acc),
    "final_test_loss": float(test_loss),
    "class_weights": {k: float(v) for k, v in class_weight_dict.items()}
}

info_path = os.path.join(OUTPUT_DIR, "training_info.json")
with open(info_path, 'w') as f:
    json.dump(training_info, f, indent=2)
print(f"   Info: {info_path}")
print()

print("=" * 70)
print("✨ TRAINING COMPLETE!")
print("=" * 70)
print()
print("🎉 Your new balanced model is ready to use!")
print()
print("Next steps:")
print("   1. Test the new model with your Flask/Streamlit apps")
print("   2. Update model path in src/model_loader.py")
print("   3. Run predictions and compare with old model")
print()
