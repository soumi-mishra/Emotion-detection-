import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import pickle

# Load train and test files
train_df = pd.read_csv("train.txt", sep=";", names=["text","emotion"])
test_df = pd.read_csv("test.txt", sep=";", names=["text","emotion"])

# Combine for tokenizer fitting
texts = pd.concat([train_df['text'], test_df['text']]).astype(str).values
labels = pd.concat([train_df['emotion'], test_df['emotion']]).values

# Tokenize text
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

# Save tokenizer
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Prepare train sequences
X_train = pad_sequences(tokenizer.texts_to_sequences(train_df['text']), maxlen=100)
X_test = pad_sequences(tokenizer.texts_to_sequences(test_df['text']), maxlen=100)

# Encode labels
unique_labels = sorted(set(labels))
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

y_train = np.array([label_to_index[label] for label in train_df['emotion']])
y_test = np.array([label_to_index[label] for label in test_df['emotion']])

y_train = to_categorical(y_train, num_classes=len(unique_labels))
y_test = to_categorical(y_test, num_classes=len(unique_labels))

# Save arrays
np.save("X_train_text.npy", X_train)
np.save("y_train_text.npy", y_train)
np.save("X_test_text.npy", X_test)
np.save("y_test_text.npy", y_test)
np.save("label_map.npy", unique_labels)

print("Preprocessing complete.")
print("Training samples:", X_train.shape[0])
print("Test samples:", X_test.shape[0])