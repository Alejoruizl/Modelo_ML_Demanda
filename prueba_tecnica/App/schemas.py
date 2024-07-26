from pydantic import BaseModel

class PredictionRequest(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    # Añade aquí todas las características necesarias

class PredictionResponse(BaseModel):
    prediction: str