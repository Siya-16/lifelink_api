from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import traceback
from fastapi.middleware.cors import CORSMiddleware

# Optional: Load environment variables if needed
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

app = FastAPI()

# Allow CORS for frontend like Flutter
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can specify your Flutter app domain here for more security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schema for POST request
class DonorData(BaseModel):
    name: str
    age: int
    gender: str
    weight: float
    blood_group: str
    location: str
    medical_history: str

# Load the model and encoders at startup
try:
    model = joblib.load("model.pkl")
    le_gender = joblib.load("le_gender.pkl")
    le_blood = joblib.load("le_blood.pkl")
    le_location = joblib.load("le_location.pkl")
    le_history = joblib.load("le_history.pkl")
except Exception as e:
    print("Model or encoder loading failed:", e)
    raise e

@app.post("/predict")
async def predict(donor: DonorData):
    try:
        # Transform input features using encoders
        gender_enc = le_gender.transform([donor.gender])[0]
        blood_enc = le_blood.transform([donor.blood_group])[0]
        location_enc = le_location.transform([donor.location])[0]
        history_enc = le_history.transform([donor.medical_history])[0]

        # Feature vector for prediction
        features = [[
            donor.age,
            gender_enc,
            donor.weight,
            blood_enc,
            location_enc,
            history_enc
        ]]

        # Predict eligibility
        prediction = model.predict(features)[0]
        eligible = bool(prediction)

        return {"eligible": eligible}

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}
