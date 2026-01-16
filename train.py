import numpy as np # type: ignore
import tensorflow as tf # type: ignore
from tensorflow.keras import models, layers # type: ignore
from tensorflow.keras.datasets import mnist # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from sklearn.utils import shuffle # type: ignore

# ---------------------------
# Load MNIST dataset
# ---------------------------
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize to 0-1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Add channel dimension
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# One-hot encode labels
num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Shuffle training data
x_train, y_train = shuffle(x_train, y_train, random_state=42)

# ---------------------------
# Build CNN model
# ---------------------------
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.Flatten(),

    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),

    layers.Dense(num_classes, activation='softmax')
])

# ---------------------------
# Compile model
# ---------------------------
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ---------------------------
# Train model
# ---------------------------
history = model.fit(
    x_train, y_train,
    validation_split=0.1,
    epochs=10,
    batch_size=128
)

# ---------------------------
# Evaluate model
# ---------------------------
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", test_acc)

# ---------------------------
# Save trained model
# ---------------------------
model.save("mnist_model.h5")
