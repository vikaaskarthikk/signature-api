from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
import cv2

app = FastAPI()

# Load your trained model
model = tf.keras.models.load_model("signature_model.h5")  # use your .h5 file
IMG_SIZE = 128

@app.get("/")
def home():
    return {"status": "Signature Verification API running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read uploaded image
    image_bytes = await file.read()
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Preprocess (same as training)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    pred = model.predict(img)[0][0]
    result = "Forged" if pred > 0.5 else "Real"

    return {
        "prediction": result,
        "confidence": float(pred)
    }
