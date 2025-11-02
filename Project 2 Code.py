
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

# Step 1: Data Processing

# The input image size for the model
IMG_SIZE = (500, 500, 3)
BATCH_SIZE = 32

# Create ImageDataGenerators
# Train data generator includes augmentation (rescale, shear, zoom)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2
)

# Validation and test generators only apply rescaling
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Create train, validation, and test data generators
# The directory names correspond to the dataset folders 
train_data = train_datagen.flow_from_directory(
    'train',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_data = val_datagen.flow_from_directory(
    'valid',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    'test',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Display class names and sample counts
print("Class indices:", train_data.class_indices)
print(f"Train samples: {train_data.samples} | Validation samples: {val_data.samples} | Test samples: {test_data.samples}")

# Optional: visualize a few training images
x_batch, y_batch = next(train_data)
plt.figure(figsize=(10, 5))
for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(x_batch[i])
    plt.axis('off')
plt.suptitle("Sample Training Images", fontsize=14)
plt.tight_layout()
plt.show()
