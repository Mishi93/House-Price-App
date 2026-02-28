import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


# Load dataset
data = pd.read_csv("Housing.csv")

# -------------------------
# Encode Binary Columns
# -------------------------

binary_cols = [
    'mainroad',
    'guestroom',
    'basement',
    'hotwaterheating',
    'airconditioning',
    'prefarea'
]

for col in binary_cols:
    data[col] = data[col].map({'no': 0, 'yes': 1})

# Encode furnishingstatus
data['furnishingstatus'] = data['furnishingstatus'].map({
    'unfurnished': 0,
    'semi-furnished': 1,
    'furnished': 2
})

# -------------------------
# Features & Target
# -------------------------

X = data[['area', 
          'bedrooms', 
          'bathrooms', 
          'stories', 
          'parking', 
          'mainroad',
          'guestroom',
          'basement',
          'hotwaterheating',
          'airconditioning',
          'prefarea',
          'furnishingstatus']]

y = data['price']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "house_price_model.joblib")

print("Model trained and saved successfully!")