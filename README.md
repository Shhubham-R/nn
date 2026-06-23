# Deep Learning Laboratory Workbook

Comprehensive Implementation of Neural Network Architectures and Algorithms

---

## Program Index

| No. | Program | Tools |
|-----|---------|-------|
| 1.a | Single-Layer Perceptron — OR, AND, XOR logic gates | Python, NumPy |
| 1.b | Multi-Layer Neural Network via Keras — XOR classification | Python, Keras, TensorFlow |
| 2 | CNN for CIFAR-10 Image Classification | Python, TensorFlow, Keras |
| 3 | MLP for MNIST Handwritten Digit Classification | Python, TensorFlow, Keras |
| 4 | Simple RNN for Sequence Modeling | Python, Keras, NumPy |
| 5 | LSTM for IMDB Sentiment Analysis | Python, TensorFlow, Keras |
| 6 | GAN for Handwritten Digit Synthesis | Python, TensorFlow, Keras, Matplotlib |
| 7 | Saliency Map Visualization | Python, TensorFlow (GradientTape), MobileNetV2 |
| 8 | Encoder-Decoder Image Captioning (CNN + LSTM) | Python, TensorFlow, InceptionV3 |

---

## Program 1.a — Single-Layer Perceptron from Scratch

**Objective:** Implement a single-layer artificial neuron from scratch utilizing fundamental NumPy matrix logic. The objective is to map linear separations and visually inspect iterative modifications to parameters across multiple evaluation iterations for logic truth tables (AND, OR, XOR).

```python
import numpy as np

features = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

and_labels = np.array([0, 0, 0, 1])
or_labels  = np.array([0, 1, 1, 1])
xor_labels = np.array([0, 1, 1, 0])

def step_function(net_input):
    return np.where(net_input >= 0, 1, 0)

def train_perceptron(features, labels, operator, max_epochs=5, alpha=0.1):
    print(f"\n--- Training Perceptron for Logical '{operator.upper()}' Gate ---")

    np.random.seed(42)
    weights = np.random.rand(2)
    bias    = np.random.rand()

    print(f"Initial Weights: {weights}")
    print(f"Initial Bias: {bias}")
    print(f"Learning Rate (alpha): {alpha}\n")

    for epoch in range(max_epochs):
        print(f"Epoch Value: {epoch + 1}")

        net_input   = np.dot(features, weights) + bias
        predictions = step_function(net_input)

        delta      = labels - predictions
        error_rate = np.mean(np.abs(delta))

        for j in range(len(features)):
            print(f"  {features[j][0]} {operator} {features[j][1]} -> Actual: "
                  f"{labels[j]} ; Predicted: {predictions[j]} "
                  f"(w1: {weights[0]:.7f} & w2: {weights[1]:.7f})")

        if error_rate == 0:
            print(">>> Convergence achieved: Zero error rate reached.")
            break

        weights += alpha * np.dot(features.T, delta)
        bias    += alpha * np.sum(delta)

for op, lbls in [("or", or_labels), ("and", and_labels), ("xor", xor_labels)]:
    train_perceptron(features, lbls, op, max_epochs=5)
```

**Expected Output:**

```
--- Training Perceptron for Logical 'OR' Gate ---
Initial Weights: [0.3745401 0.9507143]
Initial Bias: 0.73199394
Learning Rate (alpha): 0.1

Epoch Value: 1
  0 or 0 -> Actual: 0 ; Predicted: 1 (w1: 0.3745401 & w2: 0.9507143)
  0 or 1 -> Actual: 1 ; Predicted: 1 (w1: 0.3745401 & w2: 0.9507143)
  1 or 0 -> Actual: 1 ; Predicted: 1 (w1: 0.3745401 & w2: 0.9507143)
  1 or 1 -> Actual: 1 ; Predicted: 1 (w1: 0.3745401 & w2: 0.9507143)
Epoch Value: 2
  0 or 0 -> Actual: 0 ; Predicted: 0 (w1: 0.0798963 & w2: 0.2866930)
  0 or 1 -> Actual: 1 ; Predicted: 1 (w1: 0.0798963 & w2: 0.2866930)
  1 or 0 -> Actual: 1 ; Predicted: 1 (w1: 0.0798963 & w2: 0.2866930)
  1 or 1 -> Actual: 1 ; Predicted: 1 (w1: 0.0798963 & w2: 0.2866930)
>>> Convergence achieved: Zero error rate reached.
```

---

## Program 1.b — Logic Gate Modeling via Keras Sequential API

**Objective:** Build a structural multilayered model with the Keras framework. The setup serves to demonstrate computational scaling limits of linear perceptrons on non-linearly separable truth functions like XOR, utilizing intermediate hidden layers.

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

**Expected Output:**

```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 dense (Dense)               (None, 2)                 6
 activation (Activation)     (None, 2)                 0
 dense_1 (Dense)             (None, 1)                 3
 activation_1 (Activation)   (None, 1)                 0
=================================================================
Total params: 9 (36.00 B)
Trainable params: 9 (36.00 B)
Non-trainable params: 0 (0.00 B)
```

---

## Program 2 — Convolutional Neural Network (CNN) for CIFAR-10 Classification

**Objective:** Build a custom Convolutional Neural Network from scratch to process multi-channel feature frames. The target task centers on categorical sorting optimizations over the 10 separate visual target groups defining the CIFAR-10 database.

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

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

model.fit(train_images, train_labels, epochs=5, batch_size=32)
```

**Expected Output:**

```
Model: "sequential_2"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d_2 (Conv2D)           (None, 30, 30, 32)        896
 max_pooling2d_1 (MaxPool)   (None, 15, 15, 32)        0
 conv2d_3 (Conv2D)           (None, 13, 13, 64)        18496
 flatten_1 (Flatten)         (None, 10816)              0
 dense_4 (Dense)             (None, 64)                692288
 dense_5 (Dense)             (None, 10)                650
=================================================================
Total params: 712,330 (2.72 MB)
Trainable params: 712,330 (2.72 MB)

Epoch 1/5 - loss: 1.4121 - accuracy: 0.4940 - 66s 41ms/step
Epoch 2/5 - loss: 1.0524 - accuracy: 0.6305 - 62s 39ms/step
Epoch 3/5 - loss: 0.9148 - accuracy: 0.6803 - 85s 41ms/step
Epoch 4/5 - loss: 0.8160 - accuracy: 0.7155 - 62s 40ms/step
Epoch 5/5 - loss: 0.7370 - accuracy: 0.7419 - 82s 40ms/step
```

---

## Program 3 — Multi-Layer Perceptron (MLP) for MNIST Digit Classification

**Objective:** Construct a feedforward fully-connected network architecture utilizing modern optimization tools to categorize handwritten digital symbols from the MNIST benchmark dataset.

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

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

plt.imshow(x_test[0], cmap='gray')
plt.show()

prediction     = model.predict(x_test[0].reshape(1, 28, 28))
predicted_label = np.argmax(prediction, axis=1)
print(f"Predicted Class Index: {predicted_label[0]}")
```

**Expected Output:**

```
Epoch 1/5 - loss: 0.2963 - accuracy: 0.9133 - 3s 2ms/step
Epoch 2/5 - loss: 0.1445 - accuracy: 0.9571 - 3s 1ms/step
Epoch 3/5 - loss: 0.1077 - accuracy: 0.9673 - 3s 1ms/step
Epoch 4/5 - loss: 0.0880 - accuracy: 0.9728 - 3s 1ms/step
Epoch 5/5 - loss: 0.0751 - accuracy: 0.9765 - 3s 1ms/step

1/1 [==============================] - 0s 54ms/step
Predicted Class Index: 7
```

---

## Program 4 — Simple Recurrent Neural Network (RNN) for Sequence Modeling

**Objective:** Build a basic sequential Recurrent Neural Network topology tracking simple step distributions to extract and reproduce hidden pattern states in time-series records.

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
print("Training Simple RNN topology... (1000 epochs)")
model.fit(x, y, epochs=1000, verbose=0)

test_input = np.array([[[70], [80], [90]]])
prediction = model.predict(test_input)
print(f"\n--- Sequence Prediction Results ---")
print(f"Input Sequence             : [70, 80, 90]")
print(f"Predicted Next Scalar      : {prediction[0][0]:.2f}")
print(f"Target Expectation Value   : 100.00")
```

**Expected Output:**

```
Training Simple RNN topology... (1000 epochs)

--- Sequence Prediction Results ---
Input Sequence             : [70, 80, 90]
Predicted Next Scalar      : 100.00
Target Expectation Value   : 100.00
```

---

## Program 5 — Long Short-Term Memory (LSTM) for IMDB Sentiment Classification

**Objective:** Implement an advanced gated recurrent Long Short-Term Memory structure tailored to ingest word vector strings for text sentiment categorization over movie review logs.

```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

max_features = 10000
maxlen       = 200
print("Loading IMDB Movie Review Text dataset...")

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test  = sequence.pad_sequences(x_test,  maxlen=maxlen)

model = models.Sequential([
    layers.Embedding(max_features, 128, input_length=maxlen),

    layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2),

    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("Training text sequences...")
model.fit(x_train, y_train, batch_size=32, epochs=3, validation_data=(x_test, y_test))

loss, acc = model.evaluate(x_test, y_test)
print(f"\nFinal Evaluation Text Accuracy: {acc * 100:.2f}%")
```

**Expected Output:**

```
Loading IMDB Movie Review Text dataset...
Training text sequences...

Epoch 1/3
782/782 [......] - 121s 153ms/step - loss: 0.4755 - accuracy: 0.7740 - val_loss: 0.3689 - val_acc: 0.84

Epoch 2/3
782/782 [.......] - 128s 164ms/step - loss: 0.3605 - accuracy: 0.8475 - val_loss: 0.3536 - val_acc: 0.8410

Epoch 3/3
782/782 [.......] - 23s 30ms/step - loss: 0.3271 - accuracy: 0.8635

Final Evaluation Text Accuracy: 86.35%
```

---

## Program 6 — Generative Adversarial Network (GAN) for Digit Synthesis

**Objective:** Formulate a twin adversarial layout composed of adversarial Generator and Discriminator neural networks to construct synthetic digital symbols matching baseline MNIST distributions.

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
        layers.Dense(1,   activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

generator     = build_generator()
discriminator = build_discriminator()

discriminator.trainable = False
gan_input  = layers.Input(shape=(100,))
fake_image = generator(gan_input)
gan_output = discriminator(fake_image)
gan = tf.keras.Model(gan_input, gan_output)
gan.compile(optimizer='adam', loss='binary_crossentropy')

def train_gan(epochs=10000, batch_size=128):
    for epoch in range(epochs):
        idx       = np.random.randint(0, x_train.shape[0], batch_size)
        real_imgs = x_train[idx]

        noise     = np.random.normal(0, 1, (batch_size, 100))
        fake_imgs = generator.predict(noise, verbose=0)

        d_loss_real = discriminator.train_on_batch(real_imgs, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_imgs, np.zeros((batch_size, 1)))
        d_loss      = 0.5 * np.add(d_loss_real, d_loss_fake)

        noise  = np.random.normal(0, 1, (batch_size, 100))
        g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

        if epoch % 100 == 0:
            print(f"Epoch {epoch} | D Loss: {d_loss[0]:.4f} | G Loss: {g_loss:.4f}")

def plot_generated_images(generator, examples=10, dim=(1, 10), figsize=(15, 2)):
    noise            = np.random.normal(0, 1, (examples, 100))
    generated_images = generator.predict(noise, verbose=0)
    generated_images = 0.5 * generated_images + 0.5

    fig = plt.figure(figsize=figsize)
    for i in range(examples):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generated_images[i, :, :, 0], cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

train_gan(epochs=10000, batch_size=128)
```

**Expected Output:**

```
Epoch 8000 | D Loss: 0.3018 | G Loss: 2.8684
Epoch 8100 | D Loss: 0.3754 | G Loss: 3.7991
Epoch 8200 | D Loss: 0.3442 | G Loss: 2.5761
Epoch 8300 | D Loss: 0.2248 | G Loss: 2.7918
Epoch 8400 | D Loss: 0.4892 | G Loss: 2.2142
Epoch 8500 | D Loss: 0.3454 | G Loss: 2.9711
Epoch 8600 | D Loss: 0.2852 | G Loss: 2.6342
Epoch 8700 | D Loss: 0.4047 | G Loss: 2.2838
Epoch 8800 | D Loss: 0.2824 | G Loss: 2.7995
Epoch 8900 | D Loss: 0.3589 | G Loss: 2.1381
Epoch 9000 | D Loss: 0.2928 | G Loss: 2.9089
```

---

## Program 7 — Saliency Map Visualization for Model Interpretability

**Objective:** Incorporate partial derivative tracking across model weights to map output gradients directly onto source image pixel grids, isolating key activation areas driving predictions.

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

model = MobileNetV2(weights='imagenet')

img_path = 'elephant.jpg'

img        = image.load_img(img_path, target_size=(224, 224))
img_array  = image.img_to_array(img)
img_tensor = np.expand_dims(img_array, axis=0)
img_tensor = preprocess_input(img_tensor)

preds     = model.predict(img_tensor)
class_idx = np.argmax(preds[0])
print(f"Top predicted class: {decode_predictions(preds, top=1)[0]}")

img_tensor = tf.convert_to_tensor(img_tensor)

with tf.GradientTape() as tape:
    tape.watch(img_tensor)
    predictions = model(img_tensor)
    loss        = predictions[:, class_idx]

grads    = tape.gradient(loss, img_tensor)
saliency = tf.reduce_max(tf.abs(grads), axis=-1)[0]

saliency = (saliency - tf.reduce_min(saliency)) / \
           (tf.reduce_max(saliency) - tf.reduce_min(saliency))

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(saliency, cmap='hot')
plt.title("Saliency Map Activation")
plt.axis('off')

plt.show()
```

**Expected Output:**

```
1/1 [==============================] - 1s 540ms/step
Top predicted class: [('n02504458', 'African_elephant', 0.84231)]
```

---

## Program 8 — Encoder-Decoder System for Image Captioning

**Objective:** Design and implement an Encoder-Decoder system where a CNN acts as the encoder (extracting image features) and an LSTM acts as the decoder (generating a descriptive sentence).

```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import InceptionV3

def build_encoder():
    base_model    = InceptionV3(weights='imagenet')
    encoder_model = models.Model(
        inputs=base_model.input,
        outputs=base_model.get_layer('avg_pool').output
    )
    encoder_model.trainable = False
    return encoder_model

def build_caption_model(vocab_size, max_length):
    image_input = layers.Input(shape=(2048,))
    img_feat    = layers.Dropout(0.5)(image_input)
    img_feat    = layers.Dense(256, activation='relu')(img_feat)

    text_input  = layers.Input(shape=(max_length,))
    text_embed  = layers.Embedding(vocab_size, 256, mask_zero=True)(text_input)
    text_seq    = layers.Dropout(0.5)(text_embed)
    text_seq    = layers.LSTM(256)(text_seq)

    decoder = layers.add([img_feat, text_seq])
    decoder = layers.Dense(256, activation='relu')(decoder)
    output  = layers.Dense(vocab_size, activation='softmax')(decoder)

    model = models.Model(inputs=[image_input, text_input], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

vocab_size    = 5000
max_length    = 20
encoder       = build_encoder()
caption_model = build_caption_model(vocab_size, max_length)
caption_model.summary()
```

---
