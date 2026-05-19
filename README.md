# Deep Learning Lab Programs

This repository contains Deep Learning lab programs implemented using Python, TensorFlow, Keras, NumPy, and Matplotlib.  
Each program demonstrates a different Deep Learning concept such as Perceptron, CNN, RNN, LSTM, GAN, Saliency Maps, and Encoder-Decoder architectures.

---

# Programs List

## Program 1A: Single Layer Perceptron for OR Logic Gate Using NumPy

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
