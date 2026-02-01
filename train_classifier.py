import numpy as np
import os
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

# === 1. Load Data ===
DATA_DIR = "data/processed"
X = np.load(os.path.join(DATA_DIR, "X_images.npy"))
y = np.load(os.path.join(DATA_DIR, "y.npy"))

print("📊 Label counts:", Counter(y))
print("✅ Data loaded:", X.shape, y.shape)

# === 2. Preprocess Data ===
X = X / 255.0  # Normalize
X = X.reshape((-1, 256, 256, 1))  # Add channel dim

# Remove unknown labels (-1)
X = X[y >= 0]
y = y[y >= 0]

# Split before one-hot encoding
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# One-hot encode labels
num_classes = len(np.unique(y))
y_train_cat = to_categorical(y_train, num_classes=num_classes)
y_val_cat = to_categorical(y_val, num_classes=num_classes)

print(f"✅ Split into {len(X_train)} train and {len(X_val)} val samples")

# === 3. Define Simple CNN Model ===
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),  # Slight regularization
    Dense(num_classes, activation='softmax')
])

# === 4. Compile Model ===
model.compile(
    optimizer=Adam(learning_rate=0.001),  # Faster learning
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# === 5. Train Model ===
history = model.fit(
    X_train, y_train_cat,
    epochs=15,                # Changeable
    batch_size=8,
    validation_data=(X_val, y_val_cat)
)

# === 6. Save Model ===
model.save("constellation_classifier.h5")
print("✅ Model saved as constellation_classifier.h5")

import matplotlib.pyplot as plt

# === 7. Plot Accuracy & Loss Curves ===
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("figure4_training_curves.png", dpi=300)
plt.show()

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# === 8. Evaluate on Validation/Test Set ===
y_val_pred = model.predict(X_val)
y_val_pred_classes = np.argmax(y_val_pred, axis=1)
y_true = y_val

# Classification Report (Table 1: precision, recall, f1, accuracy)
report = classification_report(y_true, y_val_pred_classes, digits=4, output_dict=True)
print("📑 Classification Report:\n", classification_report(y_true, y_val_pred_classes, digits=4))

# Save as CSV for later use
import pandas as pd
df_report = pd.DataFrame(report).transpose()
df_report.to_csv("table1_classification_report.csv")
print("✅ Table 1 saved as table1_classification_report.csv")

# === 9. Confusion Matrix (Figure 5) ===
cm = confusion_matrix(y_true, y_val_pred_classes)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.savefig("figure5_confusion_matrix.png", dpi=300)
plt.show()




