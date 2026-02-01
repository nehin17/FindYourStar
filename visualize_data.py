import os
import numpy as np
import matplotlib.pyplot as plt

# Paths
DATA_DIR = "data"
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
NAMES_PATH = os.path.join(DATA_DIR, "constellation names.txt")

# Load data
X_images = np.load(os.path.join(PROCESSED_DIR, "X_images.npy"))
X_labels = np.load(os.path.join(PROCESSED_DIR, "X_labels.npy"))
y = np.load(os.path.join(PROCESSED_DIR, "y.npy"))

# Load constellation names
with open(NAMES_PATH, "r") as f:
    names = [line.strip() for line in f.readlines()]

# Plot few samples
for i in range(5):
    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    
    axs[0].imshow(X_images[i], cmap="gray")
    axs[0].set_title("Star Image")
    
    axs[1].imshow(X_labels[i], cmap="gray")
    axs[1].set_title("Label Mask")
    
    class_id = y[i]
    class_name = names[class_id] if class_id >= 0 and class_id < len(names) else "Unknown"
    axs[2].text(0.5, 0.5, class_name, fontsize=14, ha="center")
    axs[2].set_title("Class")
    axs[2].axis("off")
    
    plt.tight_layout()
    plt.show()

print(f"X_images shape: {X_images.shape}")
print(f"X_labels shape: {X_labels.shape}")
print(f"y shape: {y.shape}")



