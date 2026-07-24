import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Page settings
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗"
)

# Title
st.title("🚗 Car Price Prediction App")
st.write("Enter the car details below to predict its selling price.")

# Load dataset
df = pd.read_csv("car_prediction_data.csv")

# Prepare data
df = df.drop("Car_Name", axis=1)

# Convert categorical columns into numbers
df = pd.get_dummies(
    df,
    columns=["Fuel_Type", "Seller_Type", "Transmission"],
    drop_first=True
)

# Features and target
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# User inputs
st.subheader("Enter Car Details")

year = st.number_input(
    "Year",
    min_value=1990,
    max_value=2026,
    value=2015
)

present_price = st.number_input(
    "Present Price (in Lakhs)",
    min_value=0.0,
    value=5.0
)

kms_driven = st.number_input(
    "Kilometers Driven",
    min_value=0,
    value=30000
)

fuel_type = st.selectbox(
    "Fuel Type",
    ["Petrol", "Diesel", "CNG"]
)

seller_type = st.selectbox(
    "Seller Type",
    ["Dealer", "Individual"]
)

transmission = st.selectbox(
    "Transmission",
    ["Manual", "Automatic"]
)

owner = st.selectbox(
    "Number of Previous Owners",
    [0, 1, 2, 3]
)

# Prediction
if st.button("🔮 Predict Selling Price"):

    input_data = pd.DataFrame({
        "Year": [year],
        "Present_Price": [present_price],
        "Kms_Driven": [kms_driven],
        "Owner": [owner],
        "Fuel_Type_Diesel": [1 if fuel_type == "Diesel" else 0],
        "Fuel_Type_Petrol": [1 if fuel_type == "Petrol" else 0],
        "Seller_Type_Individual": [1 if seller_type == "Individual" else 0],
        "Transmission_Manual": [1 if transmission == "Manual" else 0]
    })

    # Make prediction
    prediction = model.predict(input_data)[0]

    st.success(
        f"💰 Estimated Selling Price: ₹{prediction:.2f} Lakhs"
    )
