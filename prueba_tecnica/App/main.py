from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from app.model import load_model, predict

app = FastAPI()

# Cargar el modelo y el escalador
model, scaler = load_model()

class PredictionRequest(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    # Añade aquí todas las características necesarias

class PredictionResponse(BaseModel):
    prediction: str

@app.post("/predict", response_model=PredictionResponse)
def get_prediction(request: PredictionRequest):
    data = pd.DataFrame([request.dict().values()], columns=request.dict().keys())
    prediction = predict(data, model, scaler)
    return PredictionResponse(prediction=prediction)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)