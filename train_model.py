import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv("donor_dataset.csv")

# Replace missing medical_history entries with string "None"
df['medical_history'] = df['medical_history'].fillna('None')

# Initialize LabelEncoders
le_gender = LabelEncoder()
le_blood = LabelEncoder()
le_location = LabelEncoder()
le_history = LabelEncoder()

# Fit and transform categorical columns
df['gender'] = le_gender.fit_transform(df['gender'])
df['blood_group'] = le_blood.fit_transform(df['blood_group'])
df['location'] = le_location.fit_transform(df['location'])
df['medical_history'] = le_history.fit_transform(df['medical_history'])

# Save encoders to disk for later use
joblib.dump(le_gender, 'le_gender.pkl')
joblib.dump(le_blood, 'le_blood.pkl')
joblib.dump(le_location, 'le_location.pkl')
joblib.dump(le_history, 'le_history.pkl')

# Prepare training data
X = df.drop(columns=['eligible'])
y = df['eligible']

# Train the RandomForest model
model = RandomForestClassifier()
model.fit(X, y)

# Save the trained model
joblib.dump(model, 'model.pkl')

print("Training complete, encoders and model saved successfully.")
