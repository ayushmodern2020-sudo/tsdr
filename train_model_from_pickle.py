import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical

# Load data
with open('dataset/train.pickle', 'rb') as f:
    train_data = pickle.load(f)
with open('dataset/valid.pickle', 'rb') as f:
    valid_data = pickle.load(f)
with open('dataset/test.pickle', 'rb') as f:
    test_data = pickle.load(f)

X_train, y_train = train_data['features'], train_data['labels']
X_valid, y_valid = valid_data['features'], valid_data['labels']
X_test, y_test = test_data['features'], test_data['labels']

# Normalize
X_train = X_train / 255.0
X_valid = X_valid / 255.0
X_test = X_test / 255.0

# Convert labels to categorical
num_classes = len(np.unique(y_train))
y_train = to_categorical(y_train, num_classes)
y_valid = to_categorical(y_valid, num_classes)
y_test = to_categorical(y_test, num_classes)

# Define model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
print("Training started...")
model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=10, batch_size=64)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc * 100:.2f}%")

# Save model
model.save('model/traffic_sign_model.h5')
print("Model saved to model/traffic_sign_model.h5")
