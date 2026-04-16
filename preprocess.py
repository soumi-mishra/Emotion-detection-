import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical

# Path to your Kaggle FER2013 dataset (with 'pixels' column)
csv_path = r"C:\Users\mishr\Desktop\CODE\Ai\EMOTION\fer2013.csv"

# Load dataset
df = pd.read_csv(csv_path)

X_train, y_train, X_test, y_test = [], [], [], []

for _, row in df.iterrows():
    pixels = np.array(row['pixels'].split(), dtype='float32')
    if row['Usage'] == 'Training':
        X_train.append(pixels)
        y_train.append(row['emotion'])
    elif row['Usage'] == 'PublicTest':
        X_test.append(pixels)
        y_test.append(row['emotion'])

# Convert to numpy arrays
X_train = np.array(X_train) / 255.0
X_test = np.array(X_test) / 255.0

# Reshape for CNN input
X_train = X_train.reshape(-1, 48, 48, 1)
X_test = X_test.reshape(-1, 48, 48, 1)

# One-hot encode labels
num_classes = 7
y_train = to_categorical(np.array(y_train), num_classes)
y_test = to_categorical(np.array(y_test), num_classes)

# Save arrays for training script
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

print("Preprocessing complete.")
print("Training data shape:", X_train.shape)
print("Test data shape:", X_test.shape)