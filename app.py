import pickle
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
from fastapi.responses import FileResponse



# 1. Initialize FastAPI

app = FastAPI(
    title="Medical Insurance Cost Prediction",
    description="Predict medical insurance charges using Machine Learning",
    version="1.0.0"
)


# 2. Load trained ML model

model_path = Path(__file__).parent / "regressor.pkl"

with open(model_path, "rb") as file:
    model = pickle.load(file)


# 3. Define input data structure

class InsuranceInput(BaseModel):

    age: int = Field(gt=0)
    sex: Literal["male", "female"]
    bmi: float = Field(gt=0)
    children: int = Field(ge=0)
    smoker: Literal["yes", "no"]
    region: Literal["southeast", "southwest", "northeast", "northwest"]


# 4. Define output data structure

class InsuranceOutput(BaseModel):

    predicted_charges: float


# 5. Home route

@app.get("/")
def home():
    html_file = Path(__file__).parent / "templates" / "index.html"
    return FileResponse(html_file)



# 6. Health check route

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


# 7. Prediction route

@app.post("/predict", response_model=InsuranceOutput)
def predict(data: InsuranceInput):

    try:

        # Convert categorical values into numerical values

        sex = 0 if data.sex == "male" else 1

        smoker = 0 if data.smoker == "yes" else 1

        region_mapping = {
            "southeast": 0,
            "southwest": 1,
            "northeast": 2,
            "northwest": 3
        }

        region = region_mapping[data.region]

        # Prepare input features in training order

        features = [[
            data.age,
            sex,
            data.bmi,
            data.children,
            smoker,
            region
        ]]

        # Generate prediction

        prediction = model.predict(features)

        # Return prediction

        return {
            "predicted_charges": float(prediction[0])
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )