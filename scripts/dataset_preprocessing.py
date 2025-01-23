from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Set the correct dataset directory path
dataset_dir = "C:/Users/HP/Desktop/Marathi-Sign-Language-Recognition/MSL/"

# Ensure the dataset folder exists
if not os.path.exists(dataset_dir):
    print(f"Error: Dataset directory not found at {dataset_dir}")
    exit()

# Image preprocessing and augmentation
datagen = ImageDataGenerator(
    rescale=1.0 / 255,        # Normalize pixel values
    validation_split=0.2,     # Split 20% of data for validation
    rotation_range=20,        # Augment with rotation
    width_shift_range=0.2,    # Augment with width shift
    height_shift_range=0.2,   # Augment with height shift
    horizontal_flip=True      # Augment with horizontal flip
)

# Load training data
train_data = datagen.flow_from_directory(
    dataset_dir,
    target_size=(128, 128),   # Resize images to 128x128
    batch_size=32,            # Batch size
    class_mode='categorical', # Multi-class labels
    subset="training"         # Training subset
)

# Load validation data
val_data = datagen.flow_from_directory(
    dataset_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset="validation"       # Validation subset
)

# Display class indices and dataset statistics
if __name__ == "__main__":
    print("Class Indices:")
    print(train_data.class_indices)
    print(f"\nNumber of training samples: {train_data.samples}")
    print(f"Number of validation samples: {val_data.samples}")
