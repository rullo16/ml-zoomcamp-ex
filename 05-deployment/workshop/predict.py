import pickle

import pandas as pd
import numpy as np
from fastapi import FastAPI
import uvicorn
from typing import Literal
from pydantic import BaseModel, Field

class Customer(BaseModel):
    gender: Literal["male", "female"]
    seniorcitizen: Literal[0, 1]
    partner: Literal["yes", "no"]
    dependents: Literal["yes", "no"]
    phoneservice: Literal["yes", "no"]
    multiplelines: Literal["no", "yes", "no_phone_service"]
    internetservice: Literal["dsl", "fiber_optic", "no"]
    onlinesecurity: Literal["no", "yes", "no_internet_service"]
    onlinebackup: Literal["no", "yes", "no_internet_service"]
    deviceprotection: Literal["no", "yes", "no_internet_service"]
    techsupport: Literal["no", "yes", "no_internet_service"]
    streamingtv: Literal["no", "yes", "no_internet_service"]
    streamingmovies: Literal["no", "yes", "no_internet_service"]
    contract: Literal["month-to-month", "one_year", "two_year"]
    paperlessbilling: Literal["yes", "no"]
    paymentmethod: Literal[
        "electronic_check",
        "mailed_check",
        "bank_transfer_(automatic)",
        "credit_card_(automatic)",
    ]
    tenure: int = Field(..., ge=0)
    monthlycharges: float = Field(..., ge=0.0)
    totalcharges: float = Field(..., ge=0.0)

class PredictResponse(BaseModel):
    churn_probability: float
    churn: bool

model_file = 'model.bin'

with open(model_file, 'rb') as f_in:
    scaler, ohe, model = pickle.load(f_in)

app = FastAPI(title='churn')

@app.post('/predict')
def predict(customer: Customer):

    num_f = ['tenure', 'monthlycharges', 'totalcharges']
    cat_f = ['gender',
             'seniorcitizen',
             'partner',
             'dependents',
             'phoneservice',
             'multiplelines',
             'internetservice',
             'onlinesecurity',
             'onlinebackup',
             'deviceprotection',
             'techsupport',
             'streamingtv',
             'streamingmovies',
             'contract',
             'paperlessbilling',
             'paymentmethod', ]
    customer = pd.DataFrame([customer.model_dump()])
    X = np.column_stack([scaler.transform(customer[num_f].values), ohe.transform(customer[cat_f].values)])
    y_pred = model.predict_proba(X)[:,1]
    churn = y_pred >= 0.5

    return PredictResponse(churn_probability=float(y_pred[0]), churn=bool(churn[0]))

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=9696)