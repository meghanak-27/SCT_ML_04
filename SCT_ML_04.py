import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt 

datadir = 'leapGestRecog'

def load_data(data_dir, img_size=(64, 64)):
    data = []
    labels = []
    class_names = []

    # Iterate through each main folder (01, 02, ...)
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)

        # Ensure it's a directory
        if not os.path.isdir(folder_path):
            continue
        
        # Iterate through each subfolder (01_palm, 02_thumb, ...)
        for subfolder in os.listdir(folder_path):
            subfolder_path = os.path.join(folder_path, subfolder)

            # Ensure it's a directory
            if not os.path.isdir(subfolder_path):
                continue

            class_names.append(subfolder)  # Save the subfolder name as the class label

            for img_name in os.listdir(subfolder_path):
                img_path = os.path.join(subfolder_path, img_name)

                # Skip non-image files
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')) and os.path.isfile(img_path):
                    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                    if img is not None:
                        img = cv2.resize(img, img_size)
                        data.append(img)
                        labels.append(subfolder)  # Use the subfolder name as the label

    return np.array(data), np.array(labels), class_names

# Load the data
print("Loading data...")
X, y, class_names = load_data(datadir)

# Normalize the image data
X = X.astype('float32') / 255.0

# Encode labels to integers
label_to_index = {label: index for index, label in enumerate(set(y))}
y_encoded = np.array([label_to_index[label] for label in y])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Data Augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'  

)
datagen.fit(X_train)

from tensorflow.keras.callbacks import EarlyStopping 
# Build the CNN model with regularization
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'), 

    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu', kernel_regularizer='l2'), 

    layers.Dropout(0.5),
    layers.Dense(len(label_to_index), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']) 


# Train the model with data augmentation and early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5)
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    epochs=50,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stop]) 


# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc:.4f}')

# Plot training & validation accuracy and loss values
def plot_training_history(history):
    plt.figure(figsize=(12, 4))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

# Call the plot function
plot_training_history(history)

# Function to plot images with labels
def plot_images(X, y_true, y_pred=None, class_names=None, num_images=10):
    indices = np.random.choice(len(X), num_images, replace=False)
    X_subset = X[indices]
    y_true_subset = y_true[indices]
    y_pred_subset = y_pred[indices] if y_pred is not None else None

    plt.figure(figsize=(15, 8))
    for i, (img, label) in enumerate(zip(X_subset, y_true_subset)):
        plt.subplot(2, 5, i + 1)
        plt.imshow(img)
        plt.axis('off')
        expected = f"Expected: {class_names[label]}"
        actual = f"Actual: {class_names[y_pred_subset[i]]}" if y_pred_subset is not None else "No Prediction"
        plt.title(f"{expected}\n{actual}", fontsize=10, color='blue' if y_pred_subset is None or label == y_pred_subset[i] else 'red')
    plt.tight_layout()
    plt.show()

# Make predictions on test data
y_pred = np.argmax(model.predict(X_test), axis=1)

# Display some predictions
plot_images(X_test, y_test, y_pred=y_pred, class_names=class_names)