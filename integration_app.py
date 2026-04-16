import streamlit as st
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load models
face_model = load_model("face_emotion_model.h5")
text_model = load_model("text_emotion_model.h5")

# Load tokenizer and labels for text
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
unique_labels = np.load("label_map.npy", allow_pickle=True)

# Emotion labels for face
face_emotions = ('Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral')

st.title("📊 Professional Emotion Detection Dashboard")

option = st.sidebar.selectbox("Choose input type:", ["Webcam (Face)", "Text"])

def plot_probabilities(probabilities, labels, title):
    fig, ax = plt.subplots()
    ax.bar(labels, probabilities)
    ax.set_title(title)
    ax.set_ylabel("Confidence")
    ax.set_ylim(0,1)
    st.pyplot(fig)

if option == "Text":
    sentence = st.text_input("Enter a sentence:")
    if sentence:
        seq = tokenizer.texts_to_sequences([sentence])
        padded = pad_sequences(seq, maxlen=100)
        prediction = text_model.predict(padded)[0]

        emotion = unique_labels[np.argmax(prediction)]
        st.subheader(f"Predicted Emotion: **{emotion}**")

        # Show diagnostic chart
        plot_probabilities(prediction, unique_labels, "Text Emotion Confidence Distribution")

        # Diagnostic summary
        st.write("### Diagnostic Emotional Health")
        st.write(f"- Dominant emotion: {emotion}")
        st.write(f"- Confidence: {prediction[np.argmax(prediction)]:.2f}")
        st.write(f"- Emotional balance: {len([p for p in prediction if p>0.1])} emotions above 10% confidence")

elif option == "Webcam (Face)":
    st.write("Press 'Start' to run webcam emotion detection. Press 'Stop' to exit.")
    run = st.button("Start")
    stop = st.button("Stop")

    if run and not stop:
        cap = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        stframe = st.empty()
        chart_placeholder = st.empty()
        history = []

        last_emotion = None

        while cap.isOpened() and not stop:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x,y,w,h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48,48))
                roi_gray = roi_gray.astype('float32')/255.0
                roi_gray = np.expand_dims(roi_gray, axis=0)
                roi_gray = np.expand_dims(roi_gray, axis=-1)

                prediction = face_model.predict(roi_gray)[0]
                emotion = face_emotions[np.argmax(prediction)]

                cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
                cv2.putText(frame, emotion, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

                # Update chart only if emotion changes
                if emotion != last_emotion:
                    chart_placeholder.empty()
                    chart_placeholder.subheader(f"Predicted Emotion: **{emotion}**")
                    plot_probabilities(prediction, face_emotions, "Face Emotion Confidence Distribution")

                    history.append(emotion)
                    last_emotion = emotion

            stframe.image(frame, channels="BGR")

        cap.release()

        # Show diagnostic history
        if history:
            st.write("### Emotional Health Trend")
            st.write(pd.DataFrame(history, columns=["Detected Emotion"]))