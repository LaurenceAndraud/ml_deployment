import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_iris

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with open("iris_model.pkl", "rb") as file:
    model = pickle.load(file)

app = FastAPI()

class IrisFeatures(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float

@app.get('/')
def read_root():
    return {"message": "Bienvenue sur l'API Iris"}

@app.post("/predict")
def predict(features: IrisFeatures):
    data = np.array([features.feature1, features.feature2, features.feature3, features.feature4]).reshape(1, -1)
    prediction = model.predict(data)
    return {"prediction": int(prediction[0])}

@app.get("/metrics")
def get_metrics():
    try:
        logger.info("Chargement des données Iris.")
        iris = load_iris()
        X, y = iris.data, iris.target

        logger.info("Prédictions en cours avec le modèle chargé.")
        y_pred = model.predict(X)

        logger.info("Calcul des métriques de performance.")
        accuracy = accuracy_score(y, y_pred)
        conf_matrix = confusion_matrix(y, y_pred).tolist()
        report = classification_report(y, y_pred, output_dict=True)

        rounded_report = {label: {metric: round(value, 2) for metric, value in metrics.items()} for label, metrics in report.items()}

        logger.info("Renvoi des métriques de l'API.")
        return {
            "accuracy": round(accuracy, 2),
            "confusion_matrix": conf_matrix,
            "classification_report": rounded_report
        }
    except Exception as e:
        logger.error(f"Une erreur est survenue : {e}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur")

