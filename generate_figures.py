import os
import matplotlib.pyplot as plt

# Path to dataset
data_dir = r"C:\Users\PRABHA\OneDrive\Documents\FindYourStar\backend\data\images" # adjust path
classes = os.listdir(data_dir)

counts = []
for cls in classes:
    counts.append(len(os.listdir(os.path.join(data_dir, cls))))

plt.bar(classes, counts)
plt.xlabel("Constellations")
plt.ylabel("Number of Images")
plt.title("Class Distribution After Augmentation")
plt.xticks(rotation=45)
plt.show()

