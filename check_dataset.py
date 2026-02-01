import os

# Paths
images_path = os.path.join("data", "images")
labels_path = os.path.join("data", "labels")
names_path = os.path.join("data", "constellation names.txt")

# Check images
image_files = []
for root, _, files in os.walk(images_path):
    image_files.extend([f for f in files if f.endswith(".png")])
print(f"✅ Found {len(image_files)} star images")

# Check labels
label_files = []
for root, _, files in os.walk(labels_path):
    label_files.extend([f for f in files if f.endswith(".png")])
print(f"✅ Found {len(label_files)} label images")

# Check constellation names
if os.path.exists(names_path):
    with open(names_path, "r") as f:
        names = [line.strip() for line in f.readlines()]
    print(f"✅ Found {len(names)} constellation names")
    print("Example:", names[:5])
else:
    print("⚠️ Constellation names file not found!")
