import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("text_emotion_model.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load label map
unique_labels = np.load("label_map.npy", allow_pickle=True)

# Ask user for input
sentence = input("Enter a sentence to analyze emotion: ")

# Preprocess input
seq = tokenizer.texts_to_sequences([sentence])
padded = pad_sequences(seq, maxlen=100)

# Predict emotion
prediction = model.predict(padded)
emotion = unique_labels[np.argmax(prediction)]

print("Predicted emotion:", emotion)