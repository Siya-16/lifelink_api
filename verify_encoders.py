import joblib

def verify_medical_history_encoder(encoder_path='le_history.pkl'):
    le_history = joblib.load(encoder_path)
    print("Classes in medical_history encoder:", le_history.classes_)
    if 'None' in le_history.classes_:
        print("✅ 'None' is in the encoder classes.")
    else:
        print("❌ 'None' is NOT in the encoder classes. You must retrain your model including 'None' values.")

if __name__ == "__main__":
    verify_medical_history_encoder()
