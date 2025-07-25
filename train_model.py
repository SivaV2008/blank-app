import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# Load and clean the data
df = pd.read_csv("California_Fire_Incidents.csv")
df = df.dropna(subset=['AcresBurned', 'Engines', 'PersonnelInvolved', 'Dozers',
                       'CrewsInvolved', 'Helicopters', 'AirTankers', 'WaterTenders'])
df = df[df["AcresBurned"] < 100000]

# Train
X = df[['Engines', 'PersonnelInvolved', 'Dozers', 'CrewsInvolved',
        'Helicopters', 'AirTankers', 'WaterTenders']]
y = df['AcresBurned']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save it
joblib.dump(model, "model.pkl")
print("✅ New model saved cleanly.")
