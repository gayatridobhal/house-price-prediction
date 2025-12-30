from fastapi import FastAPI
from pydantic import BaseModel #user input validation
from joblib import load
import numpy as np
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'model', 'ridge_model.joblib')

app = FastAPI() #initialize API
model = load(MODEL_PATH)

class House(BaseModel): #input expected by API
    OverallQual: int
    GrLivArea: float
    GarageCars: int
    TotalBsmtSF: float
    YearBuilt: int

@app.post("/predict")  #POST API endpoint
def predict_price(house: House): #clean python object received from JSON input
    data = pd.DataFrame([house.model_dump()])
    pred = np.expm1(model.predict(data)[0]) #fetching scalar value from array
    return{"Predicted Price:" : round(pred,2)}
