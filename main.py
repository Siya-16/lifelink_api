from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

app = FastAPI()

# Allow CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with specific Flutter web/mobile origin for security
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schema
class DonorData(BaseModel):
    name: str
    age: int
    gender: str
    weight: float
    blood_group: str
    location: str
    medical_history: str

# Load model and encoders once at startup
try:
    model = joblib.load("model.pkl")
    le_gender = joblib.load("le_gender.pkl")
    le_blood = joblib.load("le_blood.pkl")
    le_location = joblib.load("le_location.pkl")
    le_history = joblib.load("le_history.pkl")
except Exception as e:
    print(f"Model loading failed: {e}")
    model = None

@app.post("/predict")
async def predict(donor: DonorData):
    if model is None:
        return JSONResponse(status_code=500, content={"error": "Model not loaded on server"})

    try:
        # Encode features
        gender_enc = le_gender.transform([donor.gender])[0]
        blood_enc = le_blood.transform([donor.blood_group])[0]
        location_enc = le_location.transform([donor.location])[0]
        history_enc = le_history.transform([donor.medical_history])[0]

        features = [[
            donor.age,
            gender_enc,
            donor.weight,
            blood_enc,
            location_enc,
            history_enc
        ]]

        prediction = model.predict(features)[0]
        eligible = bool(prediction)

        return {"eligible": eligible}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
