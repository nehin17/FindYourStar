import numpy as np
import matplotlib.pyplot as plt
import random

# Load data
X = np.load("data/processed/X_images.npy")
y = np.load("data/processed/y.npy")
with open("data/processed/label_names.txt", "r") as f:
    label_names = [line.strip() for line in f]

label_map = {i: name for i, name in enumerate(label_names)}


# Reverse label_map to go from label index to folder
reverse_map = {v: k for k, v in label_map.items()}

# Plot 5 random samples per class
for class_idx, class_name in label_map.items():
    indices = [i for i, label in enumerate(y) if label == class_idx]
    if len(indices) == 0:
        print(f"⚠️ No samples found for class: {class_name}")
        continue

    selected = random.sample(indices, min(5, len(indices)))

    plt.figure(figsize=(16, 4))
    for i, idx in enumerate(selected):
        plt.subplot(1, 5, i + 1)
        img = X[idx]
        if img.shape[-1] == 1:  # If grayscale
            plt.imshow(img.squeeze(), cmap="gray")
        else:
            plt.imshow(img)
        plt.axis("off")
        plt.title(class_name)
    plt.suptitle(f"Label: {class_name}", fontsize=16)
    plt.savefig(f"constellation_{class_name}.png")
    print("Label map:", label_map)
    plt.close()
    


