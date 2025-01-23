import cv2
import numpy as np
import tensorflow as tf

def predict_sign(frame, model, classes):
    img = cv2.resize(frame, (128, 128))
    img = np.expand_dims(img, axis=0) / 255.0
    prediction = model.predict(img)
    return classes[np.argmax(prediction)]

def real_time_recognition():
    cap = cv2.VideoCapture(0)
    classes = list(train_data.class_indices.keys())  # Class names
    model = tf.keras.models.load_model("marathi_sign_model.h5")

    predicted_text = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        predicted_sign = predict_sign(frame, model, classes)

        if predicted_sign == "Space":
            predicted_text += " "
        elif predicted_sign == "Backspace" and len(predicted_text) > 0:
            predicted_text = predicted_text[:-1]
        else:
            predicted_text += predicted_sign

        # Display the predicted sign and the text
        cv2.putText(frame, f"Sign: {predicted_sign}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Text: {predicted_text}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Real-Time Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
