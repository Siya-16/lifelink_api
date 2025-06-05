from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables for Supabase (optional here)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

app = FastAPI()

# Allow CORS so your Flutter app can call the API (adjust origins if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict to your Flutter app origin for security
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define input schema for request body validation
class DonorData(BaseModel):
    name: str
    age: int
    gender: str
    weight: float
    blood_group: str
    location: str
    medical_history: str

# Load your model and encoders once at startup
model = joblib.load("model.pkl")
le_gender = joblib.load("le_gender.pkl")
le_blood = joblib.load("le_blood.pkl")
le_location = joblib.load("le_location.pkl")
le_history = joblib.load("le_history.pkl")

@app.post("/predict")
async def predict(donor: DonorData):
    try:
        # Encode categorical features using your label encoders
        gender_enc = le_gender.transform([donor.gender])[0]
        blood_enc = le_blood.transform([donor.blood_group])[0]
        location_enc = le_location.transform([donor.location])[0]
        history_enc = le_history.transform([donor.medical_history])[0]

        # Prepare feature vector for prediction - adjust order & features to your model!
        features = [[
            donor.age,
            gender_enc,
            donor.weight,
            blood_enc,
            location_enc,
            history_enc
        ]]

        prediction = model.predict(features)[0]  # Assuming 0 or 1 or similar

        # Convert prediction to boolean for eligibility
        eligible = bool(prediction)

        return {"eligible": eligible}

    except Exception as e:
        return {"error": str(e)}
