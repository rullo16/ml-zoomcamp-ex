import numpy as np
import onnxruntime as ort
from keras_image_helper import create_preprocessor
from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl
import uvicorn

app = FastAPI(title="clothing-classifier")

def preprocess_pytorch_style(X):
    X = X / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    X = (X - mean) / std
    X = X.transpose(0,3,1,2)

    return X.astype(np.float32)

preprocessor = create_preprocessor(
    preprocess_pytorch_style,
    target_size=(224, 224),
)

session = ort.InferenceSession(
    "clothing-model.onnx", providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

classes = [
"dress",
    "hat",
    "longsleeve",
    "outwear",
    "pants",
    "shirt",
    "shoes",
    "shorts",
    "skirt",
    "t-shirt",
]

class PredictRequest(BaseModel):
    url:HttpUrl

class PredictResponse(BaseModel):
    prediction: dict[str, float]
    top_class: str
    top_probability: float

def predict(url: str):
    X = preprocessor.from_url(str(url))
    result = session.run([output_name], {input_name: X})
    float_prediction = result[0][0].tolist()
    prediction_dict = dict(zip(classes, float_prediction))

    top_class = max(prediction_dict, key=prediction_dict.get)
    top_probability = prediction_dict[top_class]

    return prediction_dict, top_class, top_probability

@app.get("/")
def root():
    return {"message": "Clothing Classification Service"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest):
    predictions, top_class, top_probability = predict(request.url)

    return PredictResponse(
        prediction=predictions,
        top_class=top_class,
        top_probability=top_probability,
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
