from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    "Marathi-Sign-Recognition/MSL",
    target_size=(128, 128),
    batch_size=32,
    subset="training"
)

val_data = datagen.flow_from_directory(
    "Marathi-Sign-Recognition/MSL",
    target_size=(128, 128),
    batch_size=32,
    subset="validation"
)
