from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

app = FastAPI()


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


@app.get("/create_model/{name}")
async def create_model(name: str):
    try:
        model_creation_response = requests.get(f"http://localhost:8000/model/create/{name}")
        model_creation_response.raise_for_status()
        return {"create_model_response": model_creation_response.json()}

    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la création du modèle : {str(e)}")


@app.get("/fit_model/{name}")
async def fit_model(name: str):
    try:
        fit_model_response = requests.get(f"http://localhost:8000/model/fit/{name}")
        fit_model_response.raise_for_status()
        return {"fit_model_response": fit_model_response.json()}

    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'ajustement du modèle : {str(e)}")


@app.get("/predict_all/{name}")
async def predict_all(name: str):
    try:
        predict_all_response = requests.get(f"http://localhost:8000/model/predict/all/{name}")
        predict_all_response.raise_for_status()
        return {"predict_all_response": predict_all_response.json()}

    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction pour tous les exemples : {str(e)}")


@app.post("/predict/{name}")
async def predict(name: str, input_data: InputData):
    try:
        predict_response = requests.post(f"http://localhost:8000/model/predict/{name}", json=input_data.dict())
        predict_response.raise_for_status()
        return {"predict_response": predict_response.json()}

    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {str(e)}")
