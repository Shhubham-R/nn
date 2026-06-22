# Deep Learning Laboratory Workbook

## Program 1.a - Single Layer Perceptron From Scratch

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
            print(
                f"{features[j][0]} {operator} {features[j][1]} -> "
                f"Actual: {labels[j]} ; Predicted: {predictions[j]}"
            )

        if error_rate == 0:
            print(">>> Convergence achieved: Zero error rate reached.")
            break

        weights += alpha * np.dot(features.T, delta)
        bias += alpha * np.sum(delta)

for op, lbls in [
    ("or", or_labels),
    ("and", and_labels),
    ("xor", xor_labels)
]:
    train_perceptron(features, lbls, op, max_epochs=5)
```
