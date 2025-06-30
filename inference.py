# inference.py
import numpy as np
from PIL import Image
import io

def relu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)

from PIL import Image, UnidentifiedImageError
import numpy as np

def preprocess_image(file_like) -> np.ndarray:
    try:
        image = Image.open(file_like).convert("L")
        image = image.resize((28, 28))
        image_array = np.array(image).astype(np.float64)

        if np.mean(image_array) > 127:
            image_array = 255 - image_array

        image_array /= 255.0
        image_flat = image_array.reshape(784, 1)

        if image_flat.dtype != np.float64:
            raise ValueError(f"Invalid dtype: {image_flat.dtype}")
        if image_flat.shape != (784, 1):
            raise ValueError(f"Invalid shape: {image_flat.shape}")

        return image_flat

    except UnidentifiedImageError:
        raise ValueError("Invalid or unsupported image file.")
    except Exception as e:
        raise ValueError(f"Preprocessing error: {str(e)}")


def predict(X, W1, b1, W2, b2):
    Z1 = W1 @ X + b1
    A1 = relu(Z1)
    Z2 = W2 @ A1 + b2
    A2 = softmax(Z2)
    return int(np.argmax(A2, axis=0)[0]), A2
