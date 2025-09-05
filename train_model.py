import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from keras import metrics

print("🚀 Memulai training model Cat Detector...")

# Data generator dengan augmentasi untuk training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Load data dengan ukuran yang konsisten
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_gen = train_datagen.flow_from_directory(
    'datasets/train',
    target_size=IMG_SIZE,  # Konsisten dengan app.py
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    'datasets/val',
    target_size=IMG_SIZE,  # Konsisten dengan app.py
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

print(f"📊 Train samples: {train_gen.samples}")
print(f"📊 Validation samples: {val_gen.samples}")
print(f"📂 Classes: {train_gen.class_indices}")

# Model CNN yang lebih baik
model = tf.keras.Sequential([
    # Convolutional layers
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),
    
    # Dense layers
    tf.keras.layers.GlobalAveragePooling2D(),  # Lebih baik dari Flatten
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy',
             metrics.Precision(name='precision'),
             metrics.Recall(name='recall')]
)

print("🏗️ Model architecture:")
model.summary()

# Callbacks untuk training yang lebih baik
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=0.0001,
        verbose=1
    )
]

# Training
print("🎯 Memulai training...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=25,  # Lebih banyak epoch dengan early stopping
    callbacks=callbacks,
    verbose=1
)

# Evaluasi model
print("📈 Evaluating model...")
train_loss, train_acc, train_precision, train_recall = model.evaluate(train_gen, verbose=0)
val_loss, val_acc, val_precision, val_recall = model.evaluate(val_gen, verbose=0)

print(f"🎯 Training - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
print(f"✅ Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
print(f"🎯 Validation - Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")

# Simpan model
model.save('cat_model.h5')
print("✅ Model berhasil disimpan sebagai cat_model.h5")

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
print("📊 Training history saved as training_history.png")

print("🎉 Training selesai!")