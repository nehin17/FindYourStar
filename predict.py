import numpy as np
import tensorflow as tf
import cv2
import os

# ✅ Load class names (in exact order used during training: 0–6)
constellation_names = ["libra", "cassiopeia", "leo", "scorpius", "ursa major", "ursa minor", "swan"]

# ✅ Load the model
model = tf.keras.models.load_model("constellation_classifier.h5")
print("✅ Model loaded.")

# ✅ Image preprocessing function
def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))
    img = img / 255.0
    img = img.reshape(1, 256, 256, 1)
    return img

# ✅ Prediction function
def predict_constellation(img_path):
    img = preprocess_image(img_path)
    pred = model.predict(img)
    pred_class = np.argmax(pred)
    confidence = np.max(pred)

    # ✅ Safety check for index out of range
    if pred_class >= len(constellation_names):
        print(f"⚠️ Prediction index {pred_class} is out of range for loaded names.")
    else:
        print(f"🔮 Prediction: {constellation_names[pred_class]} ({confidence*100:.2f}% confidence)")

# ✅ Test with an example image
if __name__ == "__main__":
    test_path = r"C:\Users\PRABHA\OneDrive\Documents\FindYourStar\backend\data\images\libra\libra1.png"
    predict_constellation(test_path)

