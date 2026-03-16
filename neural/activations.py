
import numpy as np


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def tanh_deriv(x: np.ndarray) -> np.ndarray:
    t = np.tanh(x)
    return 1.0 - t * t


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -15, 15)
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_deriv(x: np.ndarray) -> np.ndarray:
    s = sigmoid(x)
    return s * (1.0 - s)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def relu_deriv(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float64)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / (e.sum() + 1e-8)


ACTIVATIONS = {
    "tanh": (tanh, tanh_deriv),
    "sigmoid": (sigmoid, sigmoid_deriv),
    "relu": (relu, relu_deriv),
}
