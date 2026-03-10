from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np
import joblib
import json
import os

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = joblib.load(os.path.join(BASE_DIR, "model", "mortgage_model.pkl"))
imputer = joblib.load(os.path.join(BASE_DIR, "model", "imputer.pkl"))
with open(os.path.join(BASE_DIR, "model", "feature_cols.json")) as f:
    feature_cols = json.load(f)
with open(os.path.join(BASE_DIR, "model", "cat_mappings.json")) as f:
    cat_mappings = json.load(f)

from sklearn.preprocessing import LabelEncoder
le_maps = {}
for col, cats in cat_mappings.items():
    le = LabelEncoder()
    le.fit(cats)
    le_maps[col] = le

class LoanApplication(BaseModel):
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: float
    AMT_GOODS_PRICE: float
    DAYS_BIRTH_YEARS: float
    DAYS_EMPLOYED_YEARS: float
    EXT_SOURCE_1: float
    EXT_SOURCE_2: float
    EXT_SOURCE_3: float
    CNT_CHILDREN: int
    CNT_FAM_MEMBERS: float
    REGION_RATING_CLIENT: int
    DAYS_ID_PUBLISH: float
    DAYS_REGISTRATION: float
    CODE_GENDER: str
    NAME_CONTRACT_TYPE: str
    FLAG_OWN_CAR: str
    FLAG_OWN_REALTY: str
    NAME_INCOME_TYPE: str
    NAME_EDUCATION_TYPE: str
    NAME_FAMILY_STATUS: str
    NAME_HOUSING_TYPE: str
    OCCUPATION_TYPE: str

@app.post("/api/predict")
def predict(app_data: LoanApplication):
    data = app_data.dict()
    data['DTI_RATIO'] = data['AMT_ANNUITY'] / (data['AMT_INCOME_TOTAL'] + 1e-8)
    data['CREDIT_TO_INCOME'] = min(data['AMT_CREDIT'] / (data['AMT_INCOME_TOTAL'] + 1e-8), 20)
    data['ANNUITY_TO_CREDIT'] = data['AMT_ANNUITY'] / (data['AMT_CREDIT'] + 1e-8)
    data['LTV_RATIO'] = data['AMT_CREDIT'] / (data['AMT_GOODS_PRICE'] + 1e-8)
    data['INCOME_PER_PERSON'] = data['AMT_INCOME_TOTAL'] / (data['CNT_FAM_MEMBERS'] + 1e-8)
    data['EXT_SOURCE_MEAN'] = np.mean([data['EXT_SOURCE_1'], data['EXT_SOURCE_2'], data['EXT_SOURCE_3']])

    cat_cols = ['CODE_GENDER','NAME_CONTRACT_TYPE','FLAG_OWN_CAR','FLAG_OWN_REALTY',
                'NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS',
                'NAME_HOUSING_TYPE','OCCUPATION_TYPE']
    for col in cat_cols:
        val = data[col] if data[col] in cat_mappings[col] else cat_mappings[col][0]
        data[col] = int(le_maps[col].transform([val])[0])

    row = [data.get(col, 0) for col in feature_cols]
    X = np.array(row).reshape(1, -1)

    num_cols = ['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE',
                'DAYS_BIRTH_YEARS','DAYS_EMPLOYED_YEARS','EXT_SOURCE_1','EXT_SOURCE_2',
                'EXT_SOURCE_3','EXT_SOURCE_MEAN','DTI_RATIO','CREDIT_TO_INCOME',
                'ANNUITY_TO_CREDIT','LTV_RATIO','INCOME_PER_PERSON','CNT_CHILDREN',
                'CNT_FAM_MEMBERS','REGION_RATING_CLIENT','DAYS_ID_PUBLISH','DAYS_REGISTRATION']
    num_indices = [feature_cols.index(c) for c in num_cols if c in feature_cols]
    X[0, num_indices] = imputer.transform(X[:, num_indices])[0]

    prob = float(model.predict_proba(X)[0][1])
    if prob < 0.15: risk, verdict = "LOW", "Likely Approved"
    elif prob < 0.35: risk, verdict = "MEDIUM", "Review Required"
    else: risk, verdict = "HIGH", "Likely Rejected"

    return {
        "probability": round(prob * 100, 1),
        "risk": risk,
        "verdict": verdict,
        "ratios": {
            "DTI Ratio": round(data['DTI_RATIO'] * 100, 1),
            "LTV Ratio": round(data['LTV_RATIO'], 3),
            "Credit/Income": round(data['CREDIT_TO_INCOME'], 2),
            "EXT Score": round(data['EXT_SOURCE_MEAN'], 3),
        }
    }

app.mount("/", StaticFiles(directory=os.path.join(BASE_DIR, "public"), html=True), name="static")
