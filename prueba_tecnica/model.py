import joblib
import pandas as pd

def load_model():
    model = joblib.load('model/trained_model.pkl')
    scaler = joblib.load('model/scaler.pkl')
    return model, scaler

def predict(data: pd.DataFrame, model, scaler):
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)
    return prediction[0]