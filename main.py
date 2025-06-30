# main.py
from fastapi import FastAPI, UploadFile, File
import io
from inference import preprocess_image, predict
from model import load_model

app = FastAPI()

# Load the model once
W1, b1, W2, b2 = load_model("model.pkl")

@app.post("/predict")
async def predict_digit(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image_stream = io.BytesIO(contents)
        X = preprocess_image(image_stream)
        pred, probs = predict(X, W1, b1, W2, b2)
        return {
            "prediction": pred,
            "probabilities": probs.ravel().tolist()
        }
    except Exception as e:
        return {"error": str(e)}
