import os
from PIL import Image
import numpy as np
from collections import Counter

# Define paths
DATA_DIR = "data"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
OUTPUT_DIR = os.path.join(DATA_DIR, "processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Resize target size
TARGET_SIZE = (256, 256)

# Load & resize helper
def load_and_resize(path):
    img = Image.open(path).convert("L")  # convert to grayscale
    img = img.resize(TARGET_SIZE)
    return np.array(img)

X = []
y = []
label_names = []

# Walk through folders (each folder is a constellation class)
manual_order = ["libra", "cassiopeia", "leo", "scorpius"]
for i, folder_name in enumerate(manual_order):
    folder_path = os.path.join(IMAGES_DIR, folder_name)
    if not os.path.isdir(folder_path):
        continue

    label_names.append(folder_name)  # keep mapping of label index to name

    for file in sorted(os.listdir(folder_path)):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, file)
            img_array = load_and_resize(image_path)
            X.append(img_array)
            y.append(i)

print(f"✅ Loaded {len(X)} star images from {len(label_names)} constellations")
print(f"⭐ Resized shape: {X[0].shape}")

# Save arrays
np.save(os.path.join(OUTPUT_DIR, "X_images.npy"), np.array(X))
np.save(os.path.join(OUTPUT_DIR, "y.npy"), np.array(y))
with open(os.path.join(OUTPUT_DIR, "label_names.txt"), "w") as f:
    for name in label_names:
        f.write(f"{name}\n")

print(f"💾 Saved processed data to {OUTPUT_DIR}")
print("📊 Encoded label counts:", Counter(y))

from collections import Counter
import numpy as np
import os

# Load the processed labels
y = np.load("data/processed/y.npy")

# Print label distribution
print("📊 Final label distribution:", Counter(y))

import numpy as np

y = np.load("data/processed/y.npy")
print("Unique labels in y.npy:", np.unique(y))



