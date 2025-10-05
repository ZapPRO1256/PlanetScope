from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
from fastapi.middleware.cors import CORSMiddleware
# ----------------------
# 1. Ініціалізація FastAPI
# ----------------------
app = FastAPI(title="Exoplanet Classifier API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)
# ----------------------
# 2. Загрузка моделі та скейлера
# ----------------------
model = joblib.load("exoplanet_lgb_model.pkl")
scaler = joblib.load("scaler.pkl")

# ----------------------
# 3. Схема запиту
# ----------------------
class SampleData(BaseModel):
    features: list  # список числових значень, відповідних X

# ----------------------
# 4. Ендпоінт для прогнозу
# ----------------------
@app.post("/predict")
def predict(data: SampleData):
    # Перетворюємо в numpy array та масштабування
    X = np.array([data.features])
    X_scaled = scaler.transform(X)

    # Прогноз
    y_prob = model.predict(X_scaled)
    y_label = int(y_prob[0] > 0.6)
    confidence = float(y_prob[0])

    return {
        "predicted_label": y_label,
        "confidence": confidence
    }

# ----------------------
# 5. Ендпоінт для перевірки API
# ----------------------
@app.get("/")
def root():
    return {"message": "Exoplanet Classifier API is running!"}
