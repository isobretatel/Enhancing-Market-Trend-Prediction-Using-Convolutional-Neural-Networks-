#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Retrain CNN Model with Improved Profitable Labels

This script:
1. Generates new training images using profit-based labeling
2. Splits data into train/val/test sets
3. Trains a new CNN model
4. Saves the model for simulation testing

Usage:
    python retrain_with_new_labels.py [csv_file] [first_date] [last_date]
    python retrain_with_new_labels.py EURUSD_M15.csv 2020-01-01 2023-01-01
"""

import os
import sys
import shutil
import subprocess
import splitfolders
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# ============================================================================
# Configuration
# ============================================================================
BATCH_SIZE = 64
IMG_HEIGHT = 150
IMG_WIDTH = 150
EPOCHS = 50

# ============================================================================
# GPU Configuration
# ============================================================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=7492)]
        )
    except RuntimeError as e:
        print(f"GPU configuration warning: {e}")

logical_gpus = tf.config.list_logical_devices('GPU')
print(f"{len(gpus)} Physical GPU, {len(logical_gpus)} Logical GPUs")

# ============================================================================
# Parse arguments
# ============================================================================
csv_file = sys.argv[1] if len(sys.argv) > 1 else "EURUSD_M15.csv"
first_date = sys.argv[2] if len(sys.argv) > 2 else None
last_date = sys.argv[3] if len(sys.argv) > 3 else None

# Determine paths
if not os.path.isabs(csv_file) and not os.path.exists(csv_file):
    csv_file = os.path.join('data-cache', csv_file)

pair_name = os.path.basename(csv_file).replace('.csv', '').replace('_', '')
output_images_dir = f"chart_images_profitable_{pair_name}"
split_dir = f"profitable_split_{pair_name}"
model_name = f"chart_classification_model_profitable_{pair_name}.h5"

print(f"\n{'='*60}")
print("RETRAINING WITH PROFIT-BASED LABELS")
print(f"{'='*60}")
print(f"CSV File: {csv_file}")
print(f"Date Range: {first_date or 'start'} to {last_date or 'end'}")
print(f"Output Images: {output_images_dir}")
print(f"Model Name: {model_name}")
print(f"{'='*60}\n")

# ============================================================================
# Step 1: Generate new labeled images
# ============================================================================
print("\n[STEP 1] Generating images with profit-based labels...")

if os.path.exists(output_images_dir):
    response = input(f"Directory {output_images_dir} exists. Delete and regenerate? (y/n): ")
    if response.lower() == 'y':
        shutil.rmtree(output_images_dir)
    else:
        print("Using existing images...")

if not os.path.exists(output_images_dir):
    cmd = [sys.executable, "generate_profitable_labels.py", csv_file, output_images_dir]
    if first_date:
        cmd.append(first_date)
    if last_date:
        cmd.append(last_date)
    
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print("Error generating images!")
        sys.exit(1)

# Check image counts
uptrend_count = len(os.listdir(os.path.join(output_images_dir, 'uptrend'))) if os.path.exists(os.path.join(output_images_dir, 'uptrend')) else 0
downtrend_count = len(os.listdir(os.path.join(output_images_dir, 'downtrend'))) if os.path.exists(os.path.join(output_images_dir, 'downtrend')) else 0

print(f"\nGenerated images - Uptrend: {uptrend_count}, Downtrend: {downtrend_count}")

if uptrend_count == 0 or downtrend_count == 0:
    print("ERROR: One or both classes have 0 images. Cannot train model.")
    sys.exit(1)

# ============================================================================
# Step 2: Split into train/val/test
# ============================================================================
print("\n[STEP 2] Splitting data into train/val/test sets...")

if os.path.exists(split_dir):
    shutil.rmtree(split_dir)

splitfolders.ratio(output_images_dir, output=split_dir, seed=42, ratio=(0.7, 0.15, 0.15))
print(f"Split complete: {split_dir}")

# ============================================================================
# Step 3: Create data generators
# ============================================================================
print("\n[STEP 3] Creating data generators...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False  # Don't flip - chart direction matters!
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(split_dir, 'train'),
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_generator = val_test_datagen.flow_from_directory(
    os.path.join(split_dir, 'val'),
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_generator = val_test_datagen.flow_from_directory(
    os.path.join(split_dir, 'test'),
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

print(f"Class indices: {train_generator.class_indices}")

# ============================================================================
# Step 4: Build CNN Model
# ============================================================================
print("\n[STEP 4] Building CNN model...")

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ============================================================================
# Step 5: Train the model
# ============================================================================
print("\n[STEP 5] Training model...")

# Calculate class weights for imbalanced data
total_samples = uptrend_count + downtrend_count
weight_uptrend = total_samples / (2 * uptrend_count)
weight_downtrend = total_samples / (2 * downtrend_count)
class_weights = {0: weight_downtrend, 1: weight_uptrend}
print(f"Class weights: {class_weights}")

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint(model_name, monitor='val_accuracy', save_best_only=True, verbose=1)
]

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    class_weight=class_weights
)

# ============================================================================
# Step 6: Evaluate on test set
# ============================================================================
print("\n[STEP 6] Evaluating on test set...")

test_loss, test_accuracy = model.evaluate(test_generator)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# ============================================================================
# Step 7: Save training history plot
# ============================================================================
print("\n[STEP 7] Saving training history plot...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy plot
axes[0].plot(history.history['accuracy'], label='Train Accuracy')
axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
axes[0].set_title('Model Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True)

# Loss plot
axes[1].plot(history.history['loss'], label='Train Loss')
axes[1].plot(history.history['val_loss'], label='Val Loss')
axes[1].set_title('Model Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
history_plot_name = f"training_history_profitable_{pair_name}.png"
plt.savefig(history_plot_name)
print(f"Training history saved to: {history_plot_name}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"Model saved: {model_name}")
print(f"Test Accuracy: {test_accuracy:.2%}")
print(f"\nTo use this model for simulation, update simulate_forex_pair.py to load:")
print(f'    model = load_model("{model_name}")')
print("="*60)

# Copy model to standard name for easy testing
shutil.copy(model_name, "chart_classification_model.h5")
print(f"\nAlso copied to: chart_classification_model.h5 (for compatibility)")
print("\nYou can now run simulation to test profitability!")

