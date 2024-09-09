import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2  # Using a lighter model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt

# Enable mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Define paths based on your folder structure
image_dir = './data/ISBI2016_ISIC_Part3B_Training_Data'
labels_file = './data/ISBI2016_ISIC_Part3B_Training_Data/ISBI2016_ISIC_Part3B_Training_GroundTruth.csv'

# Load labels from CSV
labels_df = pd.read_csv(labels_file)

# Adjust the DataFrame column names
labels_df.columns = ['filename', 'label']
labels_df['filepath'] = labels_df['filename'].apply(lambda x: os.path.join(image_dir, f"{x}.jpg"))

# Prepare ImageDataGenerator with rescaling and augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=40,
    brightness_range=[0.8, 1.2],
    validation_split=0.2
)

# Setup training and validation generators with larger batch size
train_generator = train_datagen.flow_from_dataframe(
    labels_df,
    x_col='filepath',
    y_col='label',
    target_size=(224, 224),
    batch_size=64,  # Increased batch size for faster training
    class_mode='binary',
    subset='training',
    shuffle=True
)

validation_generator = train_datagen.flow_from_dataframe(
    labels_df,
    x_col='filepath',
    y_col='label',
    target_size=(224, 224),
    batch_size=64,
    class_mode='binary',
    subset='validation',
    shuffle=True
)

# Use a lighter pre-trained model, MobileNetV2, for faster training
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze more layers of the base model
for layer in base_model.layers[:-10]:
    layer.trainable = False

# Define the model
model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid', dtype='float32')  # Ensure output remains in float32
])

# Compile the model with a lower learning rate
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Implement callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss')

# Train the Model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=20,
    class_weight={'benign': 1.0, 'malignant': 4.0},
    callbacks=[early_stopping, reduce_lr, model_checkpoint]
)

# Plot Results
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()