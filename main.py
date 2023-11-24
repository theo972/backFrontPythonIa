from enum import Enum

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

# Allow all origins in this example; you might want to restrict this in a production environment
frontUrl = os.getenv('FRONT_URL', 'http://127.0.0.1:8000')
iaServiceUrl = os.getenv('IA_SERVICE_URL', 'http://127.0.0.1:8002')

app.add_middleware(
    CORSMiddleware,
    allow_origins=frontUrl,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModelType(str, Enum):
    randomforest = "randomforest"
    logisticregression = "logisticregression"


class InputData(BaseModel):
    installment: float
    log_annual_inc: float
    dti: float
    fico: int
    revol_bal: int
    revol_util: float
    inq_last_6mths: int
    delinq_2yrs: int
    pub_rec: int


@app.get("/")
async def root():
    return {"message": "Bonjour depuis l'autre projet FastAPI!"}


@app.get("/create_model/{model_type}")
async def create_model(model_type: ModelType):
    try:
        model_creation_response = requests.get(f"{iaServiceUrl}/model/create/{model_type.value}")
        model_creation_response.raise_for_status()
        return {"create_model_response": model_creation_response.json()}

    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la création du modèle : {str(e)}")


@app.get("/fit_model/{model_type}")
async def fit_model(model_type: ModelType):
    try:
        fit_model_response = requests.get(f"{iaServiceUrl}/model/fit/{model_type.value}")
        fit_model_response.raise_for_status()
        return {"fit_model_response": fit_model_response.json()}

    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'ajustement du modèle : {str(e)}")


@app.get("/predict_all/{model_type}")
async def predict_all(model_type: ModelType):
    try:
        predict_all_response = requests.get(f"{iaServiceUrl}/model/predict/all/{model_type.value}")
        predict_all_response.raise_for_status()
        return {"predict_all_response": predict_all_response.json()}

    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction pour tous les exemples : {str(e)}")


@app.post("/predict/{model_type}")
async def predict(model_type: ModelType, input_data: InputData):
    try:
        predict_response = requests.post(f"{iaServiceUrl}/model/predict/{model_type.value}", json=input_data.dict())
        predict_response.raise_for_status()
        return {"predict_response": predict_response.json()}

    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {str(e)}")
