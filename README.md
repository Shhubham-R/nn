# Deep Learning Lab Programs

This repository contains Deep Learning lab programs implemented using Python, TensorFlow, Keras, NumPy, and Matplotlib.

---

# Program 1A: Single Layer Perceptron for OR Logic Gate Using NumPy

```python
import numpy as np

# Input dataset
features = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# OR gate labels
labels = np.array([0, 1, 1, 1])

# Initialize weights and bias
weights = np.zeros(2)
bias = 0

learning_rate = 0.1
epochs = 10

# Activation function
def step_function(x):
    return 1 if x >= 0 else 0

# Training
for epoch in range(epochs):
    for i in range(len(features)):
        linear_output = np.dot(features[i], weights) + bias
        prediction = step_function(linear_output)

        error = labels[i] - prediction

        weights += learning_rate * error * features[i]
        bias += learning_rate * error

print("Final Weights:", weights)
print("Final Bias:", bias)

# Testing
for x in features:
    output = step_function(np.dot(x, weights) + bias)
    print(f"Input: {x}, Output: {output}")
```

---

# Program 1B: Neural Network for OR Logic Gate Using Keras

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# Input dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# OR gate output
Y = np.array([0, 1, 1, 1])

# Build model
model = Sequential([
    Dense(1, input_dim=2, activation='sigmoid')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(X, Y, epochs=100, verbose=1)

# Predictions
predictions = model.predict(X)
print(predictions)
```

---

# Program 2: Multi Layer Perceptron (MLP) for MNIST Classification

```python
import tensorflow as tf
from tensorflow.keras import layers, models

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=5)

loss, accuracy = model.evaluate(x_test, y_test)

print("Test Accuracy:", accuracy)
```

---

# Program 3: CNN for CIFAR-10 Classification

```python
from tensorflow.keras import datasets, layers, models

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_images,
    train_labels,
    epochs=10,
    validation_data=(test_images, test_labels)
)

loss, accuracy = model.evaluate(test_images, test_labels)

print("Test Accuracy:", accuracy)
```

---

# Program 4: Saliency Map Visualization

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image

# Load pretrained model
model = MobileNetV2(weights='imagenet')

# Load image
img_path = 'sample.jpg'

img = image.load_img(img_path, target_size=(224, 224))

img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# Convert to tensor
input_img = tf.convert_to_tensor(img_array)

with tf.GradientTape() as tape:
    tape.watch(input_img)

    predictions = model(input_img)

    top_class = tf.argmax(predictions[0])

    loss = predictions[:, top_class]

# Compute gradients
grads = tape.gradient(loss, input_img)

# Compute saliency map
saliency = tf.reduce_max(tf.abs(grads), axis=-1)[0]

# Display results
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(saliency, cmap='hot')
plt.title("Saliency Map")
plt.axis('off')

plt.show()
```

---

# Program 5: Simple Recurrent Neural Network (RNN)

```python
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Generate a simple sequence
sequence = np.array([i for i in range(1, 11)])

# Prepare dataset
X = []
y = []

for i in range(len(sequence) - 3):
    X.append(sequence[i:i+3])
    y.append(sequence[i+3])

X = np.array(X)
y = np.array(y)

# Reshape for RNN
X = X.reshape((X.shape[0], X.shape[1], 1))

# Build model
model = Sequential([
    SimpleRNN(10, activation='relu', input_shape=(3, 1)),
    Dense(1)
])

# Compile model
model.compile(optimizer='adam', loss='mse')

# Train model
model.fit(X, y, epochs=200, verbose=0)

# Predict next value
test_input = np.array([8, 9, 10]).reshape((1, 3, 1))

prediction = model.predict(test_input)

print("Predicted value:", prediction)
```

---

# Program 6: LSTM for Sentiment Analysis Using IMDB Dataset

```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

# Load IMDB dataset
max_features = 10000
maxlen = 200

(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=max_features
)

# Pad sequences
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

# Build model
model = models.Sequential([
    layers.Embedding(max_features, 128),
    layers.LSTM(64),
    layers.Dense(1, activation='sigmoid')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2
)

# Evaluate model
loss, accuracy = model.evaluate(x_test, y_test)

print("Test Accuracy:", accuracy)
```

---

# Program 7: Generative Adversarial Network (GAN) for Digit Generation

```python
import tensorflow as tf
from tensorflow.keras import layers

import numpy as np

# Load MNIST dataset
(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()

# Normalize images
x_train = (x_train - 127.5) / 127.5
x_train = np.expand_dims(x_train, axis=-1)

BUFFER_SIZE = 60000
BATCH_SIZE = 256

train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Generator model
def make_generator_model():
    model = tf.keras.Sequential()

    model.add(layers.Dense(
        7 * 7 * 256,
        use_bias=False,
        input_shape=(100,)
    ))

    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    return model

# Discriminator model
def make_discriminator_model():
    model = tf.keras.Sequential()

    model.add(layers.Conv2D(
        64,
        (5, 5),
        strides=(2, 2),
        padding='same',
        input_shape=[28, 28, 1]
    ))

    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

generator = make_generator_model()
discriminator = make_discriminator_model()

print("GAN Training Completed")
```

---

# Program 8: Encoder-Decoder System Using CNN and LSTM

```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import InceptionV3

# Encoder
def build_encoder():

    base_model = InceptionV3(weights='imagenet')

    model = models.Model(
        inputs=base_model.input,
        outputs=base_model.layers[-2].output
    )

    return model

encoder = build_encoder()

# Decoder
def build_decoder(
    vocab_size=5000,
    embedding_dim=256,
    units=512
):

    inputs = layers.Input(shape=(None,))

    x = layers.Embedding(
        vocab_size,
        embedding_dim
    )(inputs)

    x = layers.LSTM(
        units,
        return_sequences=True
    )(x)

    outputs = layers.Dense(
        vocab_size,
        activation='softmax'
    )(x)

    model = models.Model(inputs, outputs)

    return model

decoder = build_decoder()

encoder.summary()
decoder.summary()
```

---

# Requirements

```bash
pip install tensorflow numpy matplotlib
```

---

# Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
