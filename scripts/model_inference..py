import joblib
import numpy as np

def load_model(path='trained_model.pkl'):
    return joblib.load(path)

def predict(model, new_data, scaler, pca):
    new_data_scaled = scaler.transform(new_data)
    new_data_pca = pca.transform(new_data_scaled)
    prediction = model.predict(new_data_pca)
    return prediction
