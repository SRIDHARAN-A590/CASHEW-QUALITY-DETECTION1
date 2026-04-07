import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras import layers, models, optimizers

# ========== CONFIG ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")

IMG_SIZE = (160, 160)  # MobileNet works well with 160x160
BATCH_SIZE = 8
EPOCHS = 20
MODEL_PATH = os.path.join(BASE_DIR, "cashew_model.h5")
CLASS_INDICES_PATH = os.path.join(BASE_DIR, "class_indices.json")

# ========== DATA GENERATORS ==========
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",  # good vs bad
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# Save class indices for later (so we know which index is "good"/"bad")
with open(CLASS_INDICES_PATH, "w") as f:
    json.dump(train_gen.class_indices, f, indent=2)
print("Class indices:", train_gen.class_indices)

# ========== BUILD MODEL (TRANSFER LEARNING) ==========
base_model = MobileNetV2(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False  # first, freeze base

inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ========== TRAIN TOP LAYERS ==========
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen
)

# ========== OPTIONAL: FINE-TUNING ==========
# Unfreeze some layers for better accuracy (optional, can skip initially)
base_model.trainable = True

# Freeze first N layers to avoid overfitting badly
fine_tune_at = 100  # tune last layers only
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("Starting fine-tuning...")
history_fine = model.fit(
    train_gen,
    epochs=10,
    validation_data=val_gen
)

# ========== SAVE MODEL ==========
model.save(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
print(f"Class indices saved to {CLASS_INDICES_PATH}")
