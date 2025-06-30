import pickle
import numpy as np

def load_model(path="model.pkl"):
    with open(path, "rb") as f:
        model = pickle.load(f)

    W1, b1, W2, b2 = model["W1"], model["b1"], model["W2"], model["b2"]
    return W1, b1, W2, b2