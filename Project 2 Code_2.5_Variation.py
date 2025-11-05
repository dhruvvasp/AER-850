# Step 5: Model Testing and Visualization

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
image = keras.preprocessing.image
import os

# Load saved model
model = keras.models.load_model("convnet_best1.keras")

# Class names
CLASS_NAMES = ['crack', 'missing-head', 'paint-off']

# Test image paths
paths = [
    "test/crack/test_crack.jpg",
    "test/missing-head/test_missinghead.jpg",
    "test/paint-off/test_paintoff.jpg",
]

# Loop through each test image
for path in paths:
    true_label = os.path.basename(os.path.dirname(path))
    img_name = os.path.basename(path).split('.')[0]

    # Preprocess the image
    img = image.load_img(path, target_size=(500, 500))
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    # Predict class probabilities
    probs = model.predict(arr, verbose=0)[0]
    pred_idx = np.argmax(probs)
    pred_label = CLASS_NAMES[pred_idx]

    # Create figure
    plt.figure(figsize=(6,6))
    plt.imshow(img)
    plt.axis('off')

    # Concise caption above image
    plt.title(
        f"True: {true_label} | Predicted: {pred_label} ({probs[pred_idx]*100:.1f}%)",
        fontsize=12, pad=15
    )

    # Probabilities below image
    prob_text = "\n".join([f"{cls}: {probs[i]*100:.2f}%" for i, cls in enumerate(CLASS_NAMES)])
    plt.figtext(0.1, 0.02, prob_text, fontsize=10, color='black', ha='left')

    # Save figure
    save_path = f"{img_name}_prediction.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved: {save_path}")

    plt.show()
