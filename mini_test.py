import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Settings
TARGET_SIZE = (256, 256)
SELECTED_CLASSES = ["libra", "cassiopeia", "leo", "scorpius"]  # 🔁 Try with the clearest classes
DATA_DIR = "data/images"

# Load and preprocess
X, y, label_names = [], [], []

for label, class_name in enumerate(SELECTED_CLASSES):
    path = os.path.join(DATA_DIR, class_name)
    label_names.append(class_name)

    for file in sorted(os.listdir(path)):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            img = Image.open(os.path.join(path, file)).convert("L").resize(TARGET_SIZE)
            X.append(np.array(img))
            y.append(label)


X = np.array(X).astype("float32") / 255.0
X = X.reshape((-1, 256, 256, 1))
y = np.array(y)
y_cat = to_categorical(y, num_classes=4)  # ✅ FIXED

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.2, stratify=y, random_state=42)

# Simple CNN
model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(256, 256, 1)),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(4, activation='softmax')  # ✅ FIXED
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=8)

# Plot accuracy
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.title("4-Class Accuracy")
plt.show()

