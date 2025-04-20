# 🏎️ F1 2025 Race Predictor  

A machine learning-powered dashboard to predict Formula 1 race lap times based on qualifying performance. Built with Python, Streamlit, and FastF1.  

---

## 📌 **Overview**  
This project predicts F1 2025 race lap times by analyzing historical qualifying and race data from the 2024 season. It uses a **Gradient Boosting Regressor** to model the relationship between qualifying times and median race lap times, providing interactive simulations for hypothetical scenarios.  

---

## 🔧 **How It Works**  
1. **Data Collection**  
   - Fetches qualifying and race lap times from the **first 5 races of the 2024 season** using the `fastf1` API.  
   - Processes data (e.g., calculates median race lap times per driver).  

2. **Machine Learning Model**  
   - Trains a `GradientBoostingRegressor` (from `scikit-learn`) with optimized hyperparameters.  
   - Evaluates performance using **Mean Absolute Error (MAE: ~0.3 seconds)**.  

3. **Interactive Dashboard**  
   - Adjust qualifying times via sliders in the sidebar to simulate different race outcomes.  
   - Displays ranked predictions and highlights the projected winner.  

---

## 🚀 **Key Features**  
✅ **Real-Time Predictions**  
   - Dynamically updates race forecasts based on user-adjusted qualifying inputs.  
✅ **Caching for Efficiency**  
   - Uses `fastf1.Cache` and Streamlit’s `@st.cache` to speed up data reloads.  
✅ **Transparent Model**  
   - Exposes training data, metrics, and model parameters for clarity.  

---

## 💡 **Why This Project?**  
- **For Fans**: Simulate "what-if" scenarios (e.g., how a slower qualifying lap impacts race results).  
- **For Analysts**: Demonstrates practical ML applications in sports analytics.  
- **For Developers**: Modular code structure (easy to extend with new features like tire strategies).  

---

## 🛠️ **Tech Stack**  
- **Python** (`streamlit`, `pandas`, `fastf1`, `scikit-learn`)  
- **Machine Learning**: Gradient Boosting, feature scaling (`StandardScaler`), train-test splits.  

---

## 📂 **Installation & Usage**  
1. Clone the repository:  
   ```bash
   git clone https://github.com/your-username/F1-2025-Predictor.git
