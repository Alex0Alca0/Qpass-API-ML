# -*- coding: utf-8 -*-

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import json
import pandas as pd
import xgboost


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class model_input(BaseModel):
    year: int
    visitantes: int
    mes: int
    dia: int
    turno: int
    es_wknd: int
    nombre_acceso_1: int
    nombre_acceso_2: int
    nombre_acceso_3: int
    nombre_acceso_4: int
    nombre_acceso_5: int
    nombre_acceso_6: int
    nombre_acceso_7: int
    nombre_acceso_8: int
    puerta_2: int
    puerta_3: int
    dias_1: int
    dias_2: int
    dias_3: int
    dias_4: int
    dias_5: int
    dias_6: int

class model_input_2(BaseModel):
    year:int
    month:int
    dayofmonth:int
    dayofweek:int







# loading the saved model
model = pickle.load(open('bayes.pkl', 'rb'))
model_2= pickle.load(open('time_s.pkl','rb'))


@app.post('/pred')
async def scoring_endpoint(item: model_input):
    df = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
    yhat = model.predict(df)
    return {"prediction": int(yhat)}

@app.post('/pred_time')
async def scoring_endpoint(item: model_input_2):
    forpred = {
    "year":[item.dict().values()],
    "month":[item.dict().values()],
    "dayofmonth":[item.dict().values()],
    "dayofweek":[item.dict().values()]
    }
    df = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
    df_2 = pd.DataFrame(forpred)
    yhat = model_2.predict(df)
    return {"prediction": int(yhat)}
