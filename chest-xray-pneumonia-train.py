import os
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight

# =========================
# CLEAN OUTPUT
# =========================
shutil.rmtree("output", ignore_errors=True)
os.makedirs("output", exist_ok=True)

# =========================
# SETTINGS
# =========================
DATA_DIR = "data"
IMG_SIZE = (224, 224)
SEED = 42

TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
TEST_FRAC  = 0.15

classes = ["normal", "pneumonia"]  # folder names = labels

# =========================
# BUILD FILE LIST (df)
# =========================
rows = []
for c in classes:
    class_dir = os.path.join(DATA_DIR, c)
    for fname in os.listdir(class_dir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            rows.append({"filename": os.path.join(class_dir, fname), "class": c})

df = pd.DataFrame(rows)
print(df["class"].value_counts())

# =========================
# SPLIT: train / valid / test
# =========================
train_df, temp_df = train_test_split(
    df,
    test_size=(1 - TRAIN_FRAC),
    random_state=SEED,
    stratify=df["class"]
)

val_share_of_temp = VAL_FRAC / (VAL_FRAC + TEST_FRAC)

val_df, test_df = train_test_split(
    temp_df,
    test_size=(1 - val_share_of_temp),
    random_state=SEED,
    stratify=temp_df["class"]
)

print("Train/Val/Test:", len(train_df), len(val_df), len(test_df))

test_df.to_csv("output/test_split.csv", index=False)
print("Saved: output/test_split.csv")

# =========================
# DATA GENERATORS
# =========================
train_gen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=20,
    zoom_range=0.2
)

val_gen = ImageDataGenerator(rescale=1./255)

Train = train_gen.flow_from_dataframe(
    train_df, x_col="filename", y_col="class",
    target_size=IMG_SIZE, batch_size=16,
    class_mode="categorical",
    shuffle=True, seed=SEED
)

Valid = val_gen.flow_from_dataframe(
    val_df, x_col="filename", y_col="class",
    target_size=IMG_SIZE, batch_size=16,
    class_mode="categorical",
    shuffle=False
)

Test = val_gen.flow_from_dataframe(
    test_df, x_col="filename", y_col="class",
    target_size=IMG_SIZE, batch_size=8,
    class_mode="categorical",
    shuffle=False
)

print("class_indices:", Train.class_indices)

# =========================
# CLASS WEIGHTS (BALANCED)
# =========================
# Map class name -> class index (0/1) consistent with generators
class_to_idx = Train.class_indices  # e.g. {'normal': 0, 'pneumonia': 1}

# Convert train labels to indices
y_train_idx = train_df["class"].map(class_to_idx).values

# Compute balanced weights
classes_idx = np.array(sorted(class_to_idx.values()))
weights_arr = compute_class_weight(
    class_weight="balanced",
    classes=classes_idx,
    y=y_train_idx
)

class_weight = {cls: w for cls, w in zip(classes_idx, weights_arr)}
print("class_weight:", class_weight)

# =========================
# MODEL
# =========================
model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())

model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.4))
model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.3))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# =========================
# CALLBACKS: EARLY STOP + SAVE BEST
# =========================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    filepath="output/best_model.h5",
    monitor="val_loss",
    save_best_only=True,
    mode="min",
    verbose=1
)

# =========================
# TRAIN (WITH class_weight)
# =========================
hist = model.fit(
    Train,
    epochs=50,
    validation_data=Valid,
    callbacks=[early_stop, checkpoint],
    class_weight=class_weight
)

# Optional: save final model as well
model.save("output/last_model.h5")
print("Saved: output/last_model.h5")

# =========================
# PLOT ACCURACY
# =========================
plt.style.use("ggplot")
plt.figure(figsize=(16, 9))
plt.plot(hist.history.get('accuracy', []), label="Train Accuracy")
plt.plot(hist.history.get('val_accuracy', []), label="Valid Accuracy")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy over the Epochs")
plt.savefig("output/accuracy.png", dpi=300, bbox_inches="tight")
plt.close()

# =========================
# TEST METRICS + CONFUSION MATRIX
# =========================
test_loss, test_acc = model.evaluate(Test, verbose=1)
print("Test loss:", test_loss)
print("Test accuracy:", test_acc)

probs = model.predict(Test, verbose=1)        # (N, 2)
y_pred = np.argmax(probs, axis=1)            # 0/1
y_true = Test.classes                        # 0/1

cm = confusion_matrix(y_true, y_pred)

idx_to_class = {v: k for k, v in Test.class_indices.items()}
labels = [idx_to_class[i] for i in range(len(idx_to_class))]

report = classification_report(y_true, y_pred, target_names=labels, digits=4)

with open("output/test_metrics.txt", "w", encoding="utf-8") as f:
    f.write(f"Test loss: {test_loss:.6f}\n")
    f.write(f"Test accuracy: {test_acc:.6f}\n\n")
    f.write("Classification report:\n")
    f.write(report)
print("Saved: output/test_metrics.txt")

cm_df = pd.DataFrame(cm, index=[f"true_{x}" for x in labels], columns=[f"pred_{x}" for x in labels])
cm_df.to_csv("output/confusion_matrix.csv", index=True)
print("Saved: output/confusion_matrix.csv")

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
fig, ax = plt.subplots(figsize=(7, 6))
disp.plot(ax=ax, values_format='d')
ax.set_title("Confusion Matrix (Test)")
plt.savefig("output/confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: output/confusion_matrix.png")
