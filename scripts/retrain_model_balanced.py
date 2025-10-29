
"""
Retrain MobileNetV2 Model with Class Balancing

This script retrains the fruit ripeness classification model with:
- Proper class weighting
- Data augmentation
- Early stopping
- Model checkpointing
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# ===== Configuration =====
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0001

TRAIN_DIR = "data/fruit_ripeness_dataset/fruit_ripeness_dataset/fruit_archive/dataset/dataset/test/_clean/_split/train"
TEST_DIR = "data/fruit_ripeness_dataset/fruit_ripeness_dataset/fruit_archive/dataset/dataset/test/_clean/_split/test"

# ===== Data Augmentation =====
train_datagen = ImageDataGenerator(
    rescale=1./127.5,
    preprocessing_function=lambda x: x - 1.0,  # Normalize to [-1, 1]
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(
    rescale=1./127.5,
    preprocessing_function=lambda x: x - 1.0
)

# ===== Load Data =====
print("Loading training data...")
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

print("Loading validation data...")
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

num_classes = len(train_generator.class_indices)
print(f"\nNumber of classes: {num_classes}")
print(f"Class indices: {train_generator.class_indices}")

# ===== Calculate Class Weights =====
print("\nCalculating class weights...")
class_counts = np.bincount(train_generator.classes)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(num_classes),
    y=train_generator.classes
)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

print("\nClass distribution:")
for class_name, class_idx in train_generator.class_indices.items():
    print(f"  {class_name:20s}: {class_counts[class_idx]:5d} samples (weight: {class_weights[class_idx]:.3f})")

# ===== Build Model =====
print("\nBuilding model...")
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze base model initially
base_model.trainable = False

# Add custom top layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"\nModel compiled successfully!")
print(f"Total parameters: {model.count_params():,}")

# ===== Callbacks =====
callbacks = [
    ModelCheckpoint(
        'models/mobilenetv2_balanced_best.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]

# ===== Train Phase 1: Top Layers Only =====
print("\n" + "="*70)
print("PHASE 1: Training top layers only")
print("="*70)

history1 = model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

# ===== Train Phase 2: Fine-tune Entire Model =====
print("\n" + "="*70)
print("PHASE 2: Fine-tuning entire model")
print("="*70)

# Unfreeze base model
base_model.trainable = True

# Recompile with lower learning rate
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE/10),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history2 = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1,
    initial_epoch=history1.epoch[-1]
)

# ===== Evaluation =====
print("\n" + "="*70)
print("FINAL EVALUATION")
print("="*70)

test_loss, test_acc = model.evaluate(test_generator)
print(f"\nTest Accuracy: {test_acc*100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Save final model
model.save('models/mobilenetv2_balanced_final.keras')
print("\n✅ Model saved: models/mobilenetv2_balanced_final.keras")

print("\n🎉 Training complete!")
