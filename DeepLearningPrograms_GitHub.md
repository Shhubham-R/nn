# Deep Learning Laboratory Programs

## Program 1.a: Single-Layer Perceptron From Scratch

```python
import numpy as np

features = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

and_labels = np.array([0, 0, 0, 1])
or_labels = np.array([0, 1, 1, 1])
xor_labels = np.array([0, 1, 1, 0])

def step_function(net_input):
    return np.where(net_input >= 0, 1, 0)

def train_perceptron(features, labels, operator, max_epochs=5, alpha=0.1):
    print(f"\n--- Training Perceptron for Logical '{operator.upper()}' Gate ---")

    np.random.seed(42)
    weights = np.random.rand(2)
    bias = np.random.rand()

    print(f"Initial Weights: {weights}")
    print(f"Initial Bias: {bias}")
    print(f"Learning Rate (alpha): {alpha}\n")

    for epoch in range(max_epochs):
        print(f"Epoch Value: {epoch + 1}")

        net_input = np.dot(features, weights) + bias
        predictions = step_function(net_input)

        delta = labels - predictions
        error_rate = np.mean(np.abs(delta))

        for j in range(len(features)):
            print(f"{features[j][0]} {operator} {features[j][1]} -> Actual: {labels[j]} ; Predicted: {predictions[j]}")

        if error_rate == 0:
            print(">>> Convergence achieved: Zero error rate reached.")
            break

        weights += alpha * np.dot(features.T, delta)
        bias += alpha * np.sum(delta)

for op, lbls in [("or", or_labels), ("and", and_labels), ("xor", xor_labels)]:
    train_perceptron(features, lbls, op, max_epochs=5)
```

## Program 1.b: Logic Gate Modeling via Keras

```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

model = Sequential()

model.add(Dense(2, input_shape=(2,)))
model.add(Activation('sigmoid'))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])

model.summary()
```

## Program 2: CNN for CIFAR-10

```python
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

train_images, test_images = train_images / 255.0, test_images / 255.0

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
model.fit(train_images, train_labels, epochs=5, batch_size=32)
```

## Program 3: MLP for MNIST

```python
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

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

plt.imshow(x_test[0], cmap='gray')
plt.show()

prediction = model.predict(x_test[0].reshape(1, 28, 28))
predicted_label = np.argmax(prediction, axis=1)

print(f"Predicted Class Index: {predicted_label[0]}")
```

## Program 4: Simple RNN

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

x = np.array([
    [[10], [20], [30]],
    [[20], [30], [40]],
    [[30], [40], [50]],
    [[40], [50], [60]],
    [[50], [60], [70]],
    [[60], [70], [80]],
    [[70], [80], [90]]
])

y = np.array([40, 50, 60, 70, 80, 90, 100])

model = Sequential([
    SimpleRNN(128, activation='relu', input_shape=(3, 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

model.fit(x, y, epochs=1000, verbose=0)

test_input = np.array([[[70], [80], [90]]])
prediction = model.predict(test_input)

print(prediction[0][0])
```

## Program 5: LSTM for IMDB Sentiment Analysis

```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

max_features = 10000
maxlen = 200

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

model = models.Sequential([
    layers.Embedding(max_features, 128, input_length=maxlen),
    layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train, y_train, batch_size=32, epochs=3, validation_data=(x_test, y_test))

loss, acc = model.evaluate(x_test, y_test)

print(acc)
```

## Program 6: GAN for Digit Synthesis

```python
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()

x_train = (x_train.astype("float32") - 127.5) / 127.5
x_train = np.expand_dims(x_train, axis=-1)

def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(128, input_dim=100, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(784, activation='tanh'),
        layers.Reshape((28, 28, 1))
    ])
    return model

def build_discriminator():
    model = tf.keras.Sequential([
        layers.Flatten(input_shape=(28, 28, 1)),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
```

## Program 7: Saliency Mapping

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

model = MobileNetV2(weights='imagenet')

img_path = 'elephant.jpg'

img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)

img_tensor = np.expand_dims(img_array, axis=0)
img_tensor = preprocess_input(img_tensor)

preds = model.predict(img_tensor)
class_idx = np.argmax(preds[0])

img_tensor = tf.convert_to_tensor(img_tensor)

with tf.GradientTape() as tape:
    tape.watch(img_tensor)
    predictions = model(img_tensor)
    loss = predictions[:, class_idx]

grads = tape.gradient(loss, img_tensor)
saliency = tf.reduce_max(tf.abs(grads), axis=-1)[0]

saliency = (saliency - tf.reduce_min(saliency)) / (
    tf.reduce_max(saliency) - tf.reduce_min(saliency)
)

plt.imshow(saliency, cmap='hot')
plt.show()
```

## Program 8: Encoder Decoder Image Captioning

```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import InceptionV3

def build_encoder():
    base_model = InceptionV3(weights='imagenet')

    encoder_model = models.Model(
        inputs=base_model.input,
        outputs=base_model.get_layer('avg_pool').output
    )

    encoder_model.trainable = False
    return encoder_model

def build_caption_model(vocab_size, max_length):
    image_input = layers.Input(shape=(2048,))
    img_feat = layers.Dropout(0.5)(image_input)
    img_feat = layers.Dense(256, activation='relu')(img_feat)

    text_input = layers.Input(shape=(max_length,))
    text_embed = layers.Embedding(vocab_size, 256, mask_zero=True)(text_input)
    text_seq = layers.Dropout(0.5)(text_embed)
    text_seq = layers.LSTM(256)(text_seq)

    decoder = layers.add([img_feat, text_seq])
    decoder = layers.Dense(256, activation='relu')(decoder)

    output = layers.Dense(vocab_size, activation='softmax')(decoder)

    model = models.Model(inputs=[image_input, text_input], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model

vocab_size = 5000
max_length = 20

encoder = build_encoder()
caption_model = build_caption_model(vocab_size, max_length)

caption_model.summary()
```
