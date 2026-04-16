import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Load preprocessed data
X_train = np.load("X_train_text.npy")
y_train = np.load("y_train_text.npy")
X_test = np.load("X_test_text.npy")
y_test = np.load("y_test_text.npy")
unique_labels = np.load("label_map.npy", allow_pickle=True)

# Build LSTM model
model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=100),
    LSTM(128),
    Dropout(0.5),
    Dense(len(unique_labels), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
checkpoint = ModelCheckpoint("best_text_emotion_model.h5", save_best_only=True, monitor='val_accuracy')

# Train
history = model.fit(
    X_train, y_train,
    batch_size=64,
    epochs=10,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, checkpoint]
)

# Save final model
model.save("text_emotion_model.h5")

# Evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {acc:.2f}")