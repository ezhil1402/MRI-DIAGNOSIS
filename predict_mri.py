import numpy as np
import tensorflow as tf
import cv2
import os
import sys
from tensorflow.keras.applications.efficientnet import preprocess_input

IMG_SIZE = 224
THRESHOLD = 0.6

# -------------------------------
# 1️⃣ Load Model
# -------------------------------
try:
    model = tf.keras.models.load_model("final_mri_diagnosis_model.h5")
    print("\n✅ Model Loaded Successfully!")
except Exception as e:
    print("\n❌ Error loading model:", e)
    sys.exit()

# -------------------------------
# 2️⃣ User Input
# -------------------------------
image_path = input("\nEnter MRI image full path: ").strip()

if not os.path.exists(image_path):
    print("\n❌ File does not exist.")
    sys.exit()

# -------------------------------
# 3️⃣ Preprocess Image (FIXED)
# -------------------------------
img = cv2.imread(image_path)

if img is None:
    print("\n❌ Invalid image file.")
    sys.exit()

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

img = preprocess_input(img)

img = np.expand_dims(img, axis=0)

# -------------------------------
# 4️⃣ Predict
# -------------------------------
prediction = model.predict(img, verbose=0)
probability = float(prediction[0][0])

print("\n🔍 Processing MRI Scan...")
print(f"📊 Raw Probability: {probability:.4f}")

if probability >= THRESHOLD:
    diagnosis = "Tumor Detected"
    confidence = probability * 100
else:
    diagnosis = "No Tumor Detected"
    confidence = (1 - probability) * 100

print("\n🧠 ===============================")
print(f"🧠 Diagnosis Result : {diagnosis}")
print(f"📊 Confidence Level : {confidence:.2f}%")
print("🧠 ===============================\n")

if 0.45 <= probability <= 0.65:
    print("⚠ WARNING: Low confidence prediction.")
    print("⚠ Consider retraining model or improving dataset.\n")

print("✅ Diagnosis Completed Successfully.")