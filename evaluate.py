import numpy as np
import os
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd

# === Class names (update if you add more later) ===
class_names = ["Libra", "Cassiopeia", "Leo", "Scorpius"]

# === Load Data ===
DATA_DIR = "data/processed"
X = np.load(os.path.join(DATA_DIR, "X_images.npy"))
y = np.load(os.path.join(DATA_DIR, "y.npy"))

# Preprocess
X = X / 255.0
X = X.reshape((-1, 256, 256, 1))

# Remove unknown labels (-1)
X = X[y >= 0]
y = y[y >= 0]

# Load trained model
model = load_model("constellation_classifier.h5")

# Predict
y_pred = model.predict(X)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = y

# === Table 1 (Classification Report with class names) ===
report = classification_report(
    y_true,
    y_pred_classes,
    digits=4,
    target_names=class_names,
    output_dict=True
)

print("📑 Classification Report:\n",
      classification_report(y_true, y_pred_classes, digits=4, target_names=class_names))

df_report = pd.DataFrame(report).transpose()
df_report.to_csv("table1_classification_report.csv", index=True)
print("✅ Table 1 saved as table1_classification_report.csv")

# === Figure 5 (Confusion Matrix with class names) ===
cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()

tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45, ha="right")
plt.yticks(tick_marks, class_names)

# Label cells
for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(
            j, i, cm[i, j],
            ha="center", va="center",
            color="white" if cm[i, j] > cm.max() / 2 else "black"
        )

plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig("figure5_confusion_matrix.png", dpi=300)
plt.show()
print("✅ Confusion Matrix saved as figure5_confusion_matrix.png")

