import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("../models/marathi_sign_model.h5")

# Define the complete list of classes in your dataset
classes = [
    "अ", "आ", "इ", "ई", "उ", "ऊ", "ए", "ऐ", "ओ", "औ",
    "क", "ख", "ग", "घ", "ङ", "च", "छ", "ज", "झ", "ञ",
    "ट", "ठ", "ड", "ढ", "ण", "त", "थ", "द", "ध", "न",
    "प", "फ", "ब", "भ", "म", "य", "र", "ल", "व",
    "श", "ष", "स", "ह", "ळ", "क्ष", "ज्ञ", "Space", "Backspace"
]

# Function to predict a gesture from a video frame
def predict_sign(frame, model, classes):
    img = cv2.resize(frame, (128, 128))
    img = np.expand_dims(img, axis=0) / 255.0
    prediction = model.predict(img)
    return classes[np.argmax(prediction)]

# Real-time gesture recognition
def real_time_demo():
    cap = cv2.VideoCapture(0)
    predicted_text = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame (optional, based on your camera setup)
        frame = cv2.flip(frame, 1)

        # Predict the gesture
        predicted_sign = predict_sign(frame, model, classes)

        # Handle Space and Backspace
        if predicted_sign == "Space":
            predicted_text += " "
        elif predicted_sign == "Backspace" and len(predicted_text) > 0:
            predicted_text = predicted_text[:-1]
        else:
            predicted_text += predicted_sign

        # Display the predicted text
        cv2.putText(frame, f"Text: {predicted_text}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Marathi Sign Language Recognition", frame)

        # Quit the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the recognition demo
if __name__ == "__main__":
    real_time_demo()
