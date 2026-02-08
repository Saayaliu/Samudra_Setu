import tensorflow as tf
import os

# Get the base directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'otolith_model.keras')

# Load the model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
    print("Model output shape:", model.output_shape)
except Exception as e:
    print(f"Error loading model from {MODEL_PATH}: {e}")