import streamlit as st
import pandas as pd
import fastf1
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np

# Streamlit app setup
st.set_page_config(page_title="F1 2025 Race Predictor", layout="wide")
st.title("🏎️ F1 2025 Race Prediction Dashboard 🏆")
st.markdown("Predicting race lap times from qualifying performance")

# Enable caching
@st.cache_resource
def enable_cache():
    fastf1.Cache.enable_cache("f1_cache")

enable_cache()

# Load and process historical data
@st.cache_data
def load_training_data():
    sessions = []
    rounds_to_load = list(range(1, 6))  # First 5 races of 2024
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, round_num in enumerate(rounds_to_load):
        status_text.text(f"Loading data for Round {round_num}...")
        progress_bar.progress((i + 1) / len(rounds_to_load))
        
        try:
            # Load qualifying
            quali = fastf1.get_session(2024, round_num, "Q")
            quali.load()
            quali_laps = quali.laps.pick_quicklaps()[["Driver", "LapTime"]]
            quali_laps["QualiTime"] = quali_laps["LapTime"].dt.total_seconds()
            
            # Load race
            race = fastf1.get_session(2024, round_num, "R")
            race.load()
            race_laps = race.laps[["Driver", "LapTime", "Compound"]]
            race_laps = race_laps[race_laps["Compound"].notna()]  # Only laps with tire data
            race_laps["RaceTime"] = race_laps["LapTime"].dt.total_seconds()
            
            # Get median race lap per driver
            median_race_times = race_laps.groupby("Driver")["RaceTime"].median().reset_index()
            
            # Merge data
            merged = quali_laps.merge(median_race_times, on="Driver")
            sessions.append(merged)
        except Exception as e:
            st.warning(f"Couldn't load Round {round_num}: {str(e)}")
            continue
    
    if not sessions:
        st.error("No data loaded! Check your internet connection.")
        st.stop()
    
    return pd.concat(sessions).dropna()

train_data = load_training_data()

# Model training
@st.cache_resource
def train_model():
    X = train_data[["QualiTime"]].rename(columns={"QualiTime": "QualifyingTime (s)"})
    y = train_data["RaceTime"]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Model with optimized parameters
    model = GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.02,
        max_depth=4,
        min_samples_leaf=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    mae = mean_absolute_error(y_test, model.predict(X_test))
    
    return model, scaler, mae

model, scaler, model_mae = train_model()

# 2025 Driver Data
qualifying_2025 = pd.DataFrame({
    "Driver": [
        "Lando Norris", "Oscar Piastri", "Max Verstappen", 
        "George Russell", "Yuki Tsunoda", "Alexander Albon",
        "Charles Leclerc", "Lewis Hamilton", "Pierre Gasly", 
        "Carlos Sainz", "Fernando Alonso", "Lance Stroll"
    ],
    "Team": [
        "McLaren", "McLaren", "Red Bull", "Mercedes",
        "RB", "Williams", "Ferrari", "Mercedes",
        "Alpine", "Ferrari", "Aston Martin", "Aston Martin"
    ],
    "QualifyingTime (s)": [
        75.096, 75.180, 75.481, 75.546, 
        75.670, 75.737, 75.755, 75.973,
        75.980, 76.062, 76.453, 76.483
    ]
})

# Sidebar controls
st.sidebar.header("Model Performance")
st.sidebar.metric("Mean Absolute Error", f"{model_mae:.3f} seconds")

st.sidebar.header("Adjust Qualifying Times")
qual_times = []
for i, row in qualifying_2025.iterrows():
    time = st.sidebar.slider(
        f"{row['Driver']} ({row['Team']})",
        min_value=74.0,
        max_value=78.0,
        value=row["QualifyingTime (s)"],
        step=0.001,
        key=f"slider_{i}"
    )
    qual_times.append(time)

qualifying_2025["QualifyingTime (s)"] = qual_times

# Make predictions
X_pred = scaler.transform(qualifying_2025[["QualifyingTime (s)"]])
qualifying_2025["PredictedRaceTime (s)"] = model.predict(X_pred)

# Sort by predicted performance
qualifying_2025 = qualifying_2025.sort_values("PredictedRaceTime (s)").reset_index(drop=True)
qualifying_2025["Position"] = qualifying_2025.index + 1

# Display results
st.header("🏁 Predicted 2025 Race Performance")
st.dataframe(
    qualifying_2025[["Position", "Driver", "Team", "PredictedRaceTime (s)"]]
    .style.format({"PredictedRaceTime (s)": "{:.3f}"}),
    height=600
)

# Highlight winner
winner = qualifying_2025.iloc[0]
st.success(f"## 🏆 Predicted Winner: {winner['Driver']} ({winner['Team']}) - {winner['PredictedRaceTime (s)']:.3f}s")

# Show model details
with st.expander("Model Details"):
    st.markdown("""
    **Model Type:** Gradient Boosting Regressor  
    **Training Data:** 2024 Season (Rounds 1-5)  
    **Features Used:**  
    - Qualifying lap time (seconds)  
    
    **Model Parameters:**  
    - 300 trees  
    - Learning rate: 0.03  
    - Max depth: 4  
    - Min samples per leaf: 5  
    """)