import numpy as np
import pandas as pd
import pickle
import os


def ReLU(Z):
    return np.maximum(Z, 0)


def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A


def forward(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1 + b2)
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


def get_predictions(A2):
    return np.argmax(A2, 0)


with open("./models/model_90.pkl", "rb") as f:
    W1, b1, W2, b2 = pickle.load(f)

data = pd.read_csv("./train.csv")
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_test = data[0:10].T  # Using the first 10 datapoints for inference
Y_test = data_test[0]
X_test = data_test[1:n]
X_test = X_test / 255.0

_, _, _, A2 = forward(W1, b1, W2, b2, X_test)
predictions = get_predictions(A2)

print("Predictions: ", predictions)
print("Actual values: ", Y_test)
