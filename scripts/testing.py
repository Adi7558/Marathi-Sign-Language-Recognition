import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the trained model
model = tf.keras.models.load_model("../models/marathi_sign_model.h5")

# Path to the dataset
dataset_dir = "../MSL/"

# Preprocess validation data
datagen = ImageDataGenerator(rescale=1./255)

# Using flow_from_directory to load images and labels based on folder structure
val_data = datagen.flow_from_directory(
    dataset_dir,
    target_size=(128, 128),   # Resize images to the input size of the model
    batch_size=32,            # Batch size for evaluation
    class_mode='categorical', # Use categorical labels (since there are multiple classes)
    shuffle=False             # Do not shuffle, as we need to keep the order for evaluation
)

# Evaluate the model
loss, accuracy = model.evaluate(val_data)
print(f"Validation Loss: {loss:.2f}")
print(f"Validation Accuracy: {accuracy:.2f}")
print(f"Class Indices: {val_data.class_indices}")
