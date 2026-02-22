import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import kagglehub

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25

# -------------------------------
# 1️⃣ Download Dataset
# -------------------------------
kagglehub.login()
dataset_path = kagglehub.dataset_download(
    "navoneel/brain-mri-images-for-brain-tumor-detection"
)

print("Dataset Path:", dataset_path)

# If folder contains subfolder like brain_tumor_dataset, adjust automatically
subfolders = os.listdir(dataset_path)
if len(subfolders) == 1 and os.path.isdir(os.path.join(dataset_path, subfolders[0])):
    train_dir = os.path.join(dataset_path, subfolders[0])
else:
    train_dir = dataset_path

print("Training Directory:", train_dir)

# -------------------------------
# 2️⃣ Data Generator (FIXED)
# -------------------------------
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_data = datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

print("Class Indices:", train_data.class_indices)

# -------------------------------
# 3️⃣ Compute Class Weights
# -------------------------------
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes
)
class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)

# -------------------------------
# 4️⃣ Build Model
# -------------------------------
base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Fine-tune last 50 layers
for layer in base_model.layers[:-50]:
    layer.trainable = False
for layer in base_model.layers[-50:]:
    layer.trainable = True

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -------------------------------
# 5️⃣ Callbacks
# -------------------------------
checkpoint = ModelCheckpoint(
    "best_mri_model.h5",
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=6,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=3,
    verbose=1
)

# -------------------------------
# 6️⃣ Train
# -------------------------------
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[checkpoint, early_stop, reduce_lr]
)

# -------------------------------
# 7️⃣ Final Evaluation
# -------------------------------
val_loss, val_acc = model.evaluate(val_data)
print("\nFinal Validation Accuracy:", val_acc * 100, "%")

model.save("final_mri_diagnosis_model.h5")
print("✅ Model Saved Successfully!")