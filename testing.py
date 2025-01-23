import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the model
model = tf.keras.models.load_model("marathi_sign_model.h5")

# Preprocess the test data
test_datagen = ImageDataGenerator(rescale=1./255)
test_data = test_datagen.flow_from_directory(
    "Marathi-Sign-Recognition/MSL/Test",
    target_size=(128, 128),
    batch_size=32
)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_data)
print(f"Test Accuracy: {test_acc}")
