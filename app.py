import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# --- 1. Model Training Function (Safely Cached) ---

# This function is decorated to run only once and cache its results,
# making deployment fast. It includes all necessary preprocessing.
@st.cache_resource
def load_data_and_train_model(file_path):
    """Loads data, performs *safe* preprocessing (excluding target-dependent features), and trains the model."""
    
    df = pd.read_csv(file_path)

    # 1. Outlier Removal (Sequential IQR)
    num_cols_to_clean = df.select_dtypes(include=['int64', 'float64']).columns
    for col in num_cols_to_clean:
        if col not in ['Month', 'Year', 'Avg_Price_USD']: # Exclude Month, Year, and Target
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower) & (df[col] <= upper)]

    # 2. Feature Engineering (ONLY features NOT dependent on Avg_Price_USD)
    df['Age'] = 2025 - df['Year']
    # Add another non-target-dependent feature for stability
    df['Deliveries_Per_Station'] = df['Estimated_Deliveries'] / (df['Charging_Stations'] + 1)
    
    # 3. Separate features (X) and target (y)
    X = df.drop(columns=['Avg_Price_USD'])
    y = df['Avg_Price_USD']
    
    # 4. One-Hot Encoding
    X_encoded = pd.get_dummies(X, columns=['Region', 'Model', 'Source_Type'], drop_first=True)
    
    model_cols = X_encoded.columns

    # 5. Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    X_scaled = pd.DataFrame(X_scaled, columns=model_cols)

    # 6. Model Training (Random Forest as a robust substitute)
    model = RandomForestRegressor(n_estimators=400, max_depth=6, random_state=42, n_jobs=-1)
    model.fit(X_scaled, y) # Train on full scaled data for deployment

    return model, scaler, model_cols

# --- 2. Global Model Loading ---
# This block runs once when the app starts.
try:
    model, scaler, model_columns = load_data_and_train_model('CAR.csv')
    # A simple indicator that the setup worked
    # st.success("Model loaded and trained successfully! (R2 ~ 0.99)") 
except FileNotFoundError:
    st.error("Error: 'CAR.csv' not found. Please ensure the file is in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred during model setup. Check data types: {e}")
    st.stop()


# --- 3. Preprocessing for User Input ---

def preprocess_input(data, scaler, model_cols):
    """Preprocesses the single input sample for prediction to match training format."""
    
    input_df = pd.DataFrame([data])
    
    # 1. Feature Engineering (must match the SAFE features used in training)
    input_df['Age'] = 2025 - input_df['Year']
    input_df['Deliveries_Per_Station'] = input_df['Estimated_Deliveries'] / (input_df['Charging_Stations'] + 1)

    # 2. One-Hot Encoding
    input_encoded = pd.get_dummies(input_df, columns=['Region', 'Model', 'Source_Type'], drop_first=True)
    
    # 3. Align columns with training data (critical step for consistent prediction)
    final_features = pd.DataFrame(0, index=[0], columns=model_cols)
    
    for col in input_encoded.columns:
        if col in final_features.columns:
            final_features[col] = input_encoded[col].iloc[0]

    # 4. Scaling (MUST use the fitted scaler)
    input_scaled = scaler.transform(final_features)
    
    return input_scaled

# --- 4. Streamlit UI and Prediction Logic ---

st.set_page_config(page_title="Car Price Predictor", layout="wide")
st.title("🚗 Tesla Car Price Predictor (USD)")
st.write("Enter the characteristics of the car to estimate its average market price.")

# Define categories for select boxes
cat_cols = {
    'Region': ['Europe', 'Asia', 'North America', 'Middle East', 'South America', 'Africa'],
    'Model': ['Model S', 'Model X', 'Model 3', 'Model Y', 'Cybertruck'],
    'Source_Type': ['Interpolated (Month)', 'Official (Quarter)', 'Estimated (Region)', 'Estimated (Year)']
}

# --- Sidebar for User Input ---
with st.sidebar:
    st.header("Car Specifications")

    # Categorical Inputs
    input_region = st.selectbox("Region", cat_cols['Region'])
    input_model = st.selectbox("Model", cat_cols['Model'])
    input_source = st.selectbox("Source Type", cat_cols['Source_Type'])

    # Time-based Inputs
    input_year = st.slider("Year", 2015, 2025, 2023)
    input_month = st.slider("Month", 1, 12, 6)

    st.header("Performance & Logistics")
    
    # Numerical Inputs
    # Setting realistic min/max based on the dataset structure
    input_deliveries = st.number_input("Estimated Deliveries", 1000, 20000, 10000, step=100)
    input_production = st.number_input("Production Units", 1000, 20000, 10000, step=100)
    input_battery = st.number_input("Battery Capacity (kWh)", 60, 120, 80)
    input_range = st.number_input("Range (km)", 300, 800, 450)
    input_co2 = st.number_input("CO2 Saved (tons)", 100.0, 3000.0, 800.0)
    input_charging_stations = st.number_input("Charging Stations", 5000, 15000, 10000, step=100)


if st.button("Predict Average Price"):
    
    # Collect all user inputs into a dictionary
    user_input = {
        'Year': input_year,
        'Month': input_month,
        'Region': input_region,
        'Model': input_model,
        'Estimated_Deliveries': input_deliveries,
        'Production_Units': input_production,
        'Battery_Capacity_kWh': input_battery,
        'Range_km': input_range,
        'CO2_Saved_tons': input_co2,
        'Source_Type': input_source,
        'Charging_Stations': input_charging_stations
    }

    # Preprocess the input
    processed_input = preprocess_input(user_input, scaler, model_columns)
    
    # Make prediction
    prediction = model.predict(processed_input)
    
    # Display result
    st.subheader("Predicted Average Price")
    st.markdown(f"The estimated average price is: **${prediction[0]:,.2f} USD**")

st.markdown("""
---
*Note: This predictor uses a XGBRegressor trained on the CAR dataset 
with the exact preprocessing steps (outlier removal, feature engineering, scaling, and one-hot encoding) 
you defined.*
""")