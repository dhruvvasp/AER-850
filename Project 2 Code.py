####################################

      # AER 850: Project 2 

####################################

import os # Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Standard imports
import numpy as np
import matplotlib.pyplot as plt

# TensorFlow and Keras imports
from tensorflow import keras
layers = keras.layers
ImageDataGenerator = keras.preprocessing.image.ImageDataGenerator

# For reproducibility
np.random.seed(42)
keras.utils.set_random_seed(42)

## 2.1: Data Processing

# The input image size for the model
IMG_SIZE = (500, 500)
BATCH_SIZE = 32


# Train data generator includes augmentation (rescale, shear, zoom, horizontal flip)
train_datagen = ImageDataGenerator (
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=15,
    horizontal_flip=True
)

# Validation only apply rescaling
val_datagen = ImageDataGenerator(rescale=1./255)

# Test apply rescaling as well (ensures fair evaluation as validation)
test_datagen = ImageDataGenerator(rescale=1./255)

# Train, validation, and test data generators
# The directory names correspond to the dataset folders 

train_data = train_datagen.flow_from_directory(
    'train',
    target_size=IMG_SIZE,
    color_mode='rgb',            
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    
)

val_data = val_datagen.flow_from_directory(
    'valid',
    target_size=IMG_SIZE,
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
   
)

test_data = test_datagen.flow_from_directory(
    'test',
    target_size=IMG_SIZE,
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    
)

# Display class names and sample counts
print("Class indices:", train_data.class_indices)
print(f"Train samples: {train_data.samples} | Validation samples: {val_data.samples} | Test samples: {test_data.samples}")

# Visualizing a few training images
x_batch, y_batch = next(train_data)
plt.figure(figsize=(10, 5))
for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(x_batch[i])
    plt.axis('off')
plt.suptitle("Sample Training Images", fontsize=14)
plt.tight_layout()
plt.show()

## 2.2 and 2.3: Neural Network Architecture Training and Design

num_classes = 3  # crack, missing-head, paint-off
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="keras") # Suppress Keras warnings for cleaner output

# CNN Model 1, using Keras Sequential API with basic architecture

convnet1 = keras.Sequential([
    # Feature extractor
    layers.InputLayer(input_shape=(500, 500, 3)),

    # First Convolutional Block
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    # Second Convolutional Block
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    # Classifier head
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.6),
    layers.Dense(num_classes, activation='softmax')
])

convnet1.compile(
    optimizer=keras.optimizers.Adam(5e-4),
    loss='categorical_crossentropy',   # generators use one-hot labels
    metrics=['accuracy']
)

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    restore_best_weights=True
)

print("Starting model training for Model 1...")
hist_v1 = convnet1.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
    callbacks=[early_stop],
    verbose=1
)

# CNN Model 2, using Keras Sequential API with deeper architecture

convnet2 = keras.Sequential([
    # Feature extractor
    layers.InputLayer(input_shape=(500, 500, 3)),

    # First Convolutional Block 
    layers.Conv2D(32, (3,3), activation=None), 
    layers.LeakyReLU(alpha=0.1),
    layers.MaxPooling2D((2,2)),

    # Second Convolutional Block
    layers.Conv2D(64, (3,3), activation=None),
    layers.LeakyReLU(alpha=0.1),
    layers.MaxPooling2D((2,2)),

    # Third Convolutional Block
    layers.Conv2D(128, (3,3), activation=None),
    layers.LeakyReLU(alpha=0.1),
    layers.MaxPooling2D((2,2)),
    

    # Classifier head
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(num_classes, activation='softmax')
])

convnet2.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Starting model training for Model 2...")
hist_v2 = convnet2.fit(
    train_data,
    validation_data=val_data,
    epochs=30,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    verbose=1
)

## 2.4: Performance Evaluation and Visualization

def plot_history(hist, name):
    # Accuracy
    plt.plot(hist.history['accuracy'], label='train')
    plt.plot(hist.history['val_accuracy'], label='val')
    plt.title(f'{name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'{name}_accuracy.png')  # Save figure
    plt.show()

    # Loss
    plt.plot(hist.history['loss'], label='train')
    plt.plot(hist.history['val_loss'], label='val')
    plt.title(f'{name} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{name}_loss.png')  # Save figure
    plt.show()

# Plot learning curves
plot_history(hist_v1, "ConvNet1")
plot_history(hist_v2, "ConvNet2")

# Final test-set evaluation
print("\n=== Test Evaluation ===")
print("ConvNet-1:")
test_loss_1, test_acc_1 = convnet1.evaluate(test_data, verbose=1)
print(f"Test accuracy: {test_acc_1:.4f} | Test loss: {test_loss_1:.4f}")

print("\nConvNet-2:")
test_loss_2, test_acc_2 = convnet2.evaluate(test_data, verbose=1)
print(f"Test accuracy: {test_acc_2:.4f} | Test loss: {test_loss_2:.4f}")

# Save the best model based on validation accuracy
convnet2.save("convnet_best.keras")
