import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("face_emotion_model.h5")

# Emotion labels (same order as training)
emotions = ('Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral')

# Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Correct cropping (y:y+h, x:x+w)
        roi_gray = gray_img[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48,48))
        roi_gray = roi_gray.astype('float32') / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1)

        prediction = model.predict(roi_gray)
        emotion = emotions[np.argmax(prediction)]

        # Draw rectangle and emotion label
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
        cv2.putText(frame, emotion, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Video Emotion Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()