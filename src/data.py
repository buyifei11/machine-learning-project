import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

DATA_PATH = "data/nyc_taxi_sample.csv"

def generate_mock_data(n_samples: int = 2000) -> pd.DataFrame:
    """Generate a realistic synthetic dataset for NYC Taxi trips."""
    np.random.seed(42)
    
    # Generate coordinates around NYC
    # approximate NYC Bounding Box: Lat: 40.5 - 40.9, Lon: -74.25 - -73.7
    pickup_lats = np.random.uniform(40.6, 40.85, n_samples)
    pickup_lons = np.random.uniform(-74.05, -73.85, n_samples)
    
    # Add some distance for drops
    dropoff_lats = pickup_lats + np.random.normal(0, 0.02, n_samples)
    dropoff_lons = pickup_lons + np.random.normal(0, 0.02, n_samples)
    
    # Calculate approximate distance multiplier
    dist = np.sqrt((pickup_lats - dropoff_lats)**2 + (pickup_lons - dropoff_lons)**2) * 50
    trip_distance = np.abs(dist + np.random.normal(0, 0.5, n_samples))
    trip_distance = np.maximum(trip_distance, 0.1) # min 0.1 miles
    
    # Fare
    base_fare = 2.5
    fare_amount = base_fare + (trip_distance * 2.5) + np.random.normal(0, 1.0, n_samples)
    fare_amount = np.maximum(fare_amount, base_fare)
    
    # Total
    total_amount = fare_amount + np.random.uniform(0, 5, n_samples) # tips, tolls
    
    # Categoricals
    payment_types = np.random.choice(["Credit card", "Cash", "No charge", "Dispute"], 
                                      p=[0.65, 0.3, 0.03, 0.02], size=n_samples)
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pickup_dayofweek = np.random.choice(days, size=n_samples)
    
    df = pd.DataFrame({
        "pickup_latitude": pickup_lats,
        "pickup_longitude": pickup_lons,
        "dropoff_latitude": dropoff_lats,
        "dropoff_longitude": dropoff_lons,
        "trip_distance": trip_distance,
        "fare_amount": fare_amount,
        "total_amount": total_amount,
        "payment_type": payment_types,
        "pickup_dayofweek": pickup_dayofweek
    })
    
    # Introduce some artificial clusters for the autoencoder to pick up on
    # E.g., Credit card payments on weekends tend to have much longer trips
    mask = (df["payment_type"] == "Credit card") & (df["pickup_dayofweek"].isin(["Saturday", "Sunday"]))
    df.loc[mask, "trip_distance"] *= 2.0
    df.loc[mask, "fare_amount"] *= 1.8
    
    # Cash payments on weekdays tend to be shorter
    mask2 = (df["payment_type"] == "Cash") & (~df["pickup_dayofweek"].isin(["Saturday", "Sunday"]))
    df.loc[mask2, "trip_distance"] *= 0.6
    
    return df

def load_data(force_regenerate=False) -> pd.DataFrame:
    """Load the dataset, generating it if it doesn't exist."""
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    if not os.path.exists(DATA_PATH) or force_regenerate:
        print(f"Generating mock data and saving to {DATA_PATH}...")
        df = generate_mock_data()
        df.to_csv(DATA_PATH, index=False)
    else:
        df = pd.read_csv(DATA_PATH)
    return df

def preprocess_data(df: pd.DataFrame):
    """
    Preprocess the dataframe for the autoencoder.
    Scale numerical features and one-hot encode categorical features.
    Returns the transformed numpy array, and the fitted preprocessor list of feature names.
    """
    numerical_features = [
        "pickup_latitude", "pickup_longitude", 
        "dropoff_latitude", "dropoff_longitude",
        "trip_distance", "fare_amount", "total_amount"
    ]
    categorical_features = ["payment_type", "pickup_dayofweek"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features)
        ]
    )
    
    X_processed = preprocessor.fit_transform(df)
    
    # Get feature names after transformation
    cat_encoder = preprocessor.named_transformers_['cat']
    cat_features_out = cat_encoder.get_feature_names_out(categorical_features)
    feature_names = numerical_features + list(cat_features_out)
    
    return X_processed, feature_names
