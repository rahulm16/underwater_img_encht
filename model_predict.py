from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
# Load the saved model
# Parameters
IMG_SIZE = (256, 256)
BATCH_SIZE = 16


import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import Sequence
import numpy as np





model = load_model("underwater_enhancement_model.h5")

# Load and preprocess a single input image
def preprocess_image(img_path):
    img = load_img(img_path, target_size=IMG_SIZE)
    img = img_to_array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def enhance_image(model, img_path):
    input_img = preprocess_image(img_path)
    enhanced_img = model.predict(input_img)[0]  # Remove batch dimension
    return enhanced_img
def enhance_model(img_path):
        # Test on a user-provided image
        input_image_path = img_path
        enhanced_image = enhance_image(model, input_image_path)
        # Save the input image
        input_img_array = img_to_array(load_img(input_image_path, target_size=IMG_SIZE)) / 255.0  # Normalize for visualization
        input_image_save_path = "static/images/input.png"
        plt.imsave(input_image_save_path, input_img_array)
        print(f"Input image saved as {input_image_save_path}")
        # Save the processed image
        output_image_path = "static/images/final_result.png"

        plt.imsave(output_image_path, enhanced_image)
        plt.imsave(output_image_path, enhanced_image)
        print(f"Enhanced image saved as {output_image_path}")
