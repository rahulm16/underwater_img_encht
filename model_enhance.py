from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import tensorflow as tf
import numpy as np

# Configuration for underwater image processing
IMG_SIZE = (256, 256)
BATCH_SIZE = 16

# Load the trained underwater enhancement model
model = load_model("underwater_enhancement_model.h5")

# Prepare underwater image for enhancement
def preprocess_image(img_path):
    img = load_img(img_path, target_size=IMG_SIZE)
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Enhance underwater image using the model
def enhance_image(model, img_path):
    input_img = preprocess_image(img_path)
    enhanced_img = model.predict(input_img)[0]
    return enhanced_img

def enhance_model(img_path):
    # Process and enhance the underwater image
    enhanced_image = enhance_image(model, img_path)
    
    # Generate output path in the same directory as input
    directory = os.path.dirname(img_path)
    filename = os.path.basename(img_path)
    output_path = os.path.join(directory, 'intermediate_' + filename)
    
    # Save enhanced underwater image
    plt.imsave(output_path, enhanced_image)
    
    return output_path
