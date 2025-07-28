import numpy as np

# XOR dataset
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
Y = np.array([[0], [1], [1], [0]])

# Hyperparameters
lr = 0.1
epochs = 10000

# Initialize weights
np.random.seed(42)
W1 = np.random.randn(2, 4)  # input to hidden (2 → 4 neurons)
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 1)  # hidden to output (4 → 1)
b2 = np.zeros((1, 1))

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

# Training
for epoch in range(epochs):
    # Forward pass
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)

    # Loss
    loss = -np.mean(Y * np.log(A2 + 1e-8) + (1 - Y) * np.log(1 - A2 + 1e-8))

    # Backward pass
    dZ2 = A2 - Y
    dW2 = np.dot(A1.T, dZ2) / X.shape[0]
    db2 = np.mean(dZ2, axis=0, keepdims=True)

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_deriv(Z1)
    dW1 = np.dot(X.T, dZ1) / X.shape[0]
    db1 = np.mean(dZ1, axis=0, keepdims=True)

    # Update weights
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}, W1: {W1}, b1: {b1}, W2: {W2}, b2: {b2}")

# Final predictions
print("Predictions:")
print(np.round(A2))
