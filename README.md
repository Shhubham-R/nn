# Deep Learning Lab Programs

---

## Program 1a: Single Layer Perceptron using NumPy

```python
import numpy as np

and_features = np.array([[0,0],[0,1],[1,0],[1,1]])
or_features  = np.array([[0,0],[0,1],[1,0],[1,1]])
xor_features = np.array([[0,0],[0,1],[1,0],[1,1]])

and_label = np.array([0,0,0,1])
or_label  = np.array([0,1,1,1])
xor_label = np.array([0,1,1,0])

def step_fun(sum):
    return np.where(sum >= 0, 1, 0)

w          = np.random.rand(2)
bias       = np.random.rand()
alpha      = 0.1
max_epochs = 10

print("Weights:", w)
print("Bias:", bias)
print("Alpha:", alpha)
print("Epoch:", max_epochs)

def perceptron(features, labels, operator):
    global w, bias
    for i in range(max_epochs):
        print("In epoch value:", i+1)
        sum       = np.dot(features, w) + bias
        predicted = step_fun(sum)
        delta     = labels - predicted
        w        += alpha * np.dot(features.T, delta)
        bias     += alpha * np.sum(delta)
        errors    = np.mean(np.abs(delta))
        for j in range(len(features)):
            print(features[j][0], "", operator, "",
                  features[j][1], "-> Actual:", labels[j], ";",
                  "Predicted:", predicted[j],
                  "[w:", w[0], "bias:", w[1], "]")
        if errors == 0:
            break

while True:
    operator = str(input("\nEnter logical operator (and, or, xor, exit): "))
    if operator == "and":
        perceptron(and_features, and_label, operator)
    elif operator == "or":
        perceptron(or_features, or_label, operator)
    elif operator == "xor":
        perceptron(xor_features, xor_label, operator)
    elif operator == "exit":
        break
    else:
        print("Invalid input. Try again!")
```

---

## Program 1b: Neural Network using Keras

```python
from tensorflow.keras import models, layers
import numpy as np

x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

model = models.Sequential()
model.add(layers.Dense(2, activation='sigmoid', input_shape=(2,)))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])

model.fit(x, y, epochs=1000, verbose=0)

model.summary()
```

---

## Program 2: CNN Architecture for CIFAR-10

```python
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
model.fit(train_images, train_labels, epochs=5)
```

---

## Program 3: Multi-Layer Perceptron (MLP) for MNIST

```python
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

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

plt.imshow(x_train[0])
plt.show()

model.predict(x_test[0].reshape(1, 28, 28)).argmax(axis=1)
```

---

## Program 4: Simple Recurrent Neural Network (RNN)

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense


X = np.array([
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

print("Training RNN... (1000 epochs)")
model.fit(X, y, epochs=1000, verbose=0)

test_input = np.array([[[70], [80], [90]]])
prediction = model.predict(test_input)

print("\n--- Results ---")
print(f"Input Sequence: [70, 80, 90]")
print(f"Predicted Next Number: {prediction[0][0]:.2f}")
print(f"Target Number: 100.00")
```

---

## Program 5: LSTM Model for Sentiment Analysis (IMDB)

```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

max_features = 10000
maxlen = 200  # Cut off reviews after 200 words

print("Loading data...")
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

print("Training...")
model.fit(x_train, y_train, batch_size=32, epochs=3,
          validation_data=(x_test, y_test))

loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")
```

---

## Program 6: Generative Adversarial Network (GAN) for Synthetic Digits

```python
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()

x_train = (x_train.astype('float32') - 127.5) / 127.5

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
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

generator     = build_generator()
discriminator = build_discriminator()

discriminator.trainable = False

gan_input  = layers.Input(shape=(100,))
fake_image = generator(gan_input)
gan_output = discriminator(fake_image)

gan = tf.keras.Model(gan_input, gan_output)
gan.compile(optimizer='adam', loss='binary_crossentropy')

def train_gan(epochs=1000, batch_size=128):
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

def show_generated_images():
    noise   = np.random.normal(0, 1, (1, 100))
    gen_img = generator.predict(noise, verbose=0)
    gen_img = 0.5 * gen_img + 0.5
    plt.imshow(gen_img[0, :, :, 0], cmap='gray')
    plt.axis('off')
    plt.show()

train_gan(epochs=10000, batch_size=128)

def plot_generated_images(generator, examples=10, dim=(1, 10), figsize=(15, 2)):
    noise            = np.random.normal(0, 1, (examples, 100))
    generated_images = generator.predict(noise, verbose=0)
    generated_images = 0.5 * generated_images + 0.5

    plt.figure(figsize=figsize)
    for i in range(examples):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i, :, :, 0],
                   interpolation='nearest', cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

plot_generated_images(generator)
```

---

## Program 7: Saliency Map

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

model = MobileNetV2(weights='imagenet')

img_path  = 'elephant.jpg'   # replace with your image path
img       = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)

img_tensor = np.expand_dims(img_array, axis=0)
img_tensor = preprocess_input(img_tensor)

preds     = model.predict(img_tensor)
class_idx = np.argmax(preds[0])
print("Predicted:", decode_predictions(preds, top=1)[0])

img_tensor = tf.convert_to_tensor(img_tensor)

with tf.GradientTape() as tape:
    tape.watch(img_tensor)
    predictions = model(img_tensor)
    loss        = predictions[:, class_idx]

grads = tape.gradient(loss, img_tensor)

saliency = tf.reduce_max(tf.abs(grads), axis=-1)[0]

saliency = (saliency - tf.reduce_min(saliency)) / (
            tf.reduce_max(saliency) - tf.reduce_min(saliency))

plt.figure(figsize=(10, 4))

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

## Program 8: Encoder-Decoder (CNN Encoder + LSTM Decoder)

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
    encoder_model.trainable = False  # Keep CNN weights frozen
    return encoder_model

def build_caption_model(vocab_size, max_length):
    image_input = layers.Input(shape=(2048,))   # InceptionV3 avg_pool size
    img_feat    = layers.Dropout(0.5)(image_input)
    img_feat    = layers.Dense(256, activation='relu')(img_feat)

    text_input = layers.Input(shape=(max_length,))
    text_embed = layers.Embedding(vocab_size, 256, mask_zero=True)(text_input)
    text_seq   = layers.Dropout(0.5)(text_embed)
    text_seq   = layers.LSTM(256)(text_seq)

    decoder = layers.add([img_feat, text_seq])
    decoder = layers.Dense(256, activation='relu')(decoder)

    output = layers.Dense(vocab_size, activation='softmax')(decoder)

    model = models.Model(inputs=[image_input, text_input], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

vocab_size = 5000
max_length = 20

encoder       = build_encoder()
caption_model = build_caption_model(vocab_size, max_length)

caption_model.summary()
```