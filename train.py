import os
import pickle
from numba import jit
import numpy as np
import pandas as pd

data = pd.read_csv("./train.csv")
data.head()

data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.0

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.0
_, m_train = X_train.shape


# @jit(nopython=True)
def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2


# @jit(nopython=True)
def ReLU(Z):
    return np.maximum(Z, 0)


# @jit(nopython=True)
def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A


# @jit(nopython=True)
def forward(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1 + b2)
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


# @jit(nopython=True)
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


# @jit(nopython=True)
def backward(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2


# @jit(nopython=True)
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2


# @jit(nopython=True)
def get_predictions(A2):
    return np.argmax(A2, 0)


# @jit(nopython=True)
def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size


# @jit(nopython=True)
def meanSquaredError(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# @jit(nopython=True)
def gradient_descent(X, Y, alpha):
    W1, b1, W2, b2 = init_params()
    flag = True
    i = 0
    while flag:
        i += 1
        Z1, A1, Z2, A2 = forward(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        predictions = get_predictions(A2)

        print("Iteration: ", i)
        accuracy = get_accuracy(predictions, Y)
        print("Accuracy: ", accuracy)

        if accuracy > 0.90:
            flag = False

    return W1, b1, W2, b2


W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10)

os.makedirs("./models", exist_ok=True)

with open("./models/model_90.pkl", "wb") as f:
    pickle.dump((W1, b1, W2, b2), f)
