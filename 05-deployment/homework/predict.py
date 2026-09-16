"""Lead-scoring prediction service.

Run locally:
    uv run uvicorn predict:app --host 0.0.0.0 --port 9696

The model path can be overridden with the MODEL_PATH environment variable,
which is how the Docker image points at the model baked into the base image.
"""

import os
import pickle

from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = os.getenv("MODEL_PATH", "pipeline_v1.bin")

with open(MODEL_PATH, "rb") as f_in:
    pipeline = pickle.load(f_in)

app = FastAPI(title="lead-scoring")


class Lead(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float


@app.post("/predict")
def predict(lead: Lead):
    # DictVectorizer expects an iterable of dicts, never a single dict.
    features = [lead.model_dump()]
    probability = float(pipeline.predict_proba(features)[0, 1])
    return {"probability": probability, "converted": probability >= 0.5}


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9696)
