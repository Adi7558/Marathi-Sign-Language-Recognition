import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img # type: ignore



# Initialize the data generator with augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Input image path
input_image_path = 'C:/Users/HP/Desktop/Marathi-Sign-Language-Recognition/MSL/original_images/Backspace.jpg'  # Replace with your image name if different

# Output directory
output_dir = 'C:/Users/HP/Desktop/Marathi-Sign-Language-Recognition/MSL/Backspace'
os.makedirs(output_dir, exist_ok=True)  # Create the output folder if it doesn’t exist

# Load and process the image
img = load_img(input_image_path)  # Load the image
x = img_to_array(img)  # Convert to Numpy array
x = x.reshape((1,) + x.shape)  # Reshape for generator

# Generate and save 50 augmented images
i = 0
for batch in datagen.flow(x, batch_size=1, save_to_dir=output_dir, save_prefix='backspace', save_format='jpeg'):
    i += 1
    if i >= 50:
        break  # Stop after generating 50 images
