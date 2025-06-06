import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import os

# Dataset paths
base_dataset = r"/Users/allan/Documents/Clean_Code/cats_and_dogs_filtered"
training_dataset_path = os.path.join(base_dataset, 'train')
validation_dataset_path = os.path.join(base_dataset, 'validation')

# --- Image Data Generators ---
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# --- Data Generators ---
train_generator = train_datagen.flow_from_directory(
    training_dataset_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    validation_dataset_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# --- Calculate Class Weights ---
from sklearn.utils.class_weight import compute_class_weight
labels = train_generator.classes
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
class_weights = {i : class_weights[i] for i in range(2)}

# --- Model Definition ---
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001), input_shape=(150, 150, 3)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# --- Custom Metrics Callback ---
class MetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_generator):
        super().__init__()
        self.val_generator = val_generator
        self.val_precisions, self.val_recalls, self.val_f1s = [], [], []

    def on_epoch_end(self, epoch, logs=None):
        self.val_generator.reset()
        val_pred_prob = self.model.predict(self.val_generator)
        val_pred = (val_pred_prob > 0.5).astype(int).flatten()
        val_true = self.val_generator.classes

        val_precision = precision_score(val_true, val_pred, zero_division=0)
        val_recall = recall_score(val_true, val_pred, zero_division=0)
        val_f1 = f1_score(val_true, val_pred, zero_division=0)

        self.val_precisions.append(val_precision)
        self.val_recalls.append(val_recall)
        self.val_f1s.append(val_f1)

        print(f"— val_precision: {val_precision:.4f} — val_recall: {val_recall:.4f} — val_f1: {val_f1:.4f}")

# --- Compile the Model ---
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

# --- Train the Model ---
callback = MetricsCallback(val_generator)
early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

history = model.fit(
    train_generator,
    epochs=30,
    validation_data=val_generator,
    callbacks=[callback, early_stop],
    class_weight=class_weights
)

# --- Plot Accuracy & F1 ---
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend(); plt.title("Accuracy")

plt.subplot(1, 3, 2)
plt.plot(callback.val_precisions, label='Val Precision')
plt.plot(callback.val_recalls, label='Val Recall')
plt.legend(); plt.title("Precision & Recall")

plt.subplot(1, 3, 3)
plt.plot(callback.val_f1s, label='Val F1 Score')
plt.legend(); plt.title("F1 Score")
plt.tight_layout()
plt.show()
