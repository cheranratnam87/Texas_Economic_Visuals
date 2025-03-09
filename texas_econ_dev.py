import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

# ---- 1️⃣ Load Data ----
st.set_page_config(layout="wide")
st.title("🏛️ Texas Economic Dashboard")
st.subheader("Explore Texas Economic Trends and Forecasts")

@st.cache_data
def load_data():
    url = "https://data.texas.gov/resource/karz-jr5v.csv"
    df = pd.read_csv(url)
    df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
    df.set_index('date', inplace=True)
    return df

df = load_data()

# ---- 2️⃣ Create Tabs ----
tab1, tab2 = st.tabs(["📈 Data Exploration", "📊 Forecasts"])

# ---- Sidebar (Now Works for Both Tabs) ----
if "forecast_months" not in st.session_state:
    st.session_state["forecast_months"] = 24  # Default to 2 years (24 months)

if st.session_state.get("active_tab", "Data Exploration") == "Forecasts":
    forecast_years = st.sidebar.slider("📅 Forecast Horizon (Years)", 1, 5, 2)
    st.session_state["forecast_months"] = forecast_years * 12  # Store value globally

# ---- TAB 1: Data Exploration ----
with tab1:
    st.session_state["active_tab"] = "Data Exploration"
    
    st.header("📈 Explore Economic Indicators")
    st.markdown("Use the filters below to explore economic trends across years and states.")

    # Year Range Selector (Sliding Bar)
    min_year, max_year = int(df.index.year.min()), int(df.index.year.max())
    selected_year_range = st.slider("Select Year Range", min_year, max_year, (min_year, max_year))

    # State Indicator Dropdown (Default: Unemployment TX)
    state_columns = [col for col in df.columns if 'nonfarm_employment' in col or 'unemployment' in col or 'consumer_confidence' in col]
    selected_state = st.selectbox("Select State Indicator", state_columns, index=state_columns.index('unemployment_tx'))

    # Filter dataset based on selection
    filtered_df = df[(df.index.year >= selected_year_range[0]) & (df.index.year <= selected_year_range[1])]

    # Time-Series Plot
    st.subheader(f"{selected_state.replace('_', ' ').title()} from {selected_year_range[0]} to {selected_year_range[1]}")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(filtered_df.index, filtered_df[selected_state], marker='o', linestyle='-')
    ax.set_ylabel(selected_state.replace('_', ' ').title())
    ax.set_title(f"Trend of {selected_state.replace('_', ' ').title()} ({selected_year_range[0]} - {selected_year_range[1]})")
    st.pyplot(fig)

    # Correlation Heatmap (Improved)
    st.subheader("📌 Correlation Heatmap of Economic Indicators")
    numeric_cols = df.select_dtypes(include=['float64']).columns
    corr_matrix = filtered_df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(corr_matrix, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5, ax=ax,
                annot_kws={"size": 8})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
    st.pyplot(fig)

    # Data Table
    st.subheader(f"📋 Data Table ({selected_year_range[0]} - {selected_year_range[1]})")
    st.dataframe(filtered_df)

# ---- TAB 2: Forecasts ----
with tab2:
    st.session_state["active_tab"] = "Forecasts"
    st.header("📊 Forecasting Texas Economic Indicators")

    # Retrieve forecast_months from session state
    forecast_months = st.session_state["forecast_months"]

    # ---- ARIMA Forecast (Unemployment) ----
    st.subheader("📉 ARIMA Forecast - Unemployment Rate (TX)")

    # Fit ARIMA Model
    arima_model = ARIMA(df['unemployment_tx'], order=(1,1,1)).fit()
    forecast_arima = arima_model.forecast(steps=forecast_months)

    # Plot ARIMA Forecast
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df.index, df['unemployment_tx'], label="Actual", color="blue")
    ax.plot(pd.date_range(df.index[-1], periods=forecast_months, freq='M'), forecast_arima, linestyle="dashed", color="orange", label="Forecast")
    ax.set_title("Unemployment Rate Forecast (TX)")
    ax.legend()
    st.pyplot(fig)

    # ---- Explanation for Unemployment Forecast ----
    st.markdown("### **🔍 Key Insights**")
    st.markdown(
        """
        **Historical Trends:**
        - **2008-2010:** Unemployment rose significantly due to the **Great Recession**.
        - **2020 Spike:** Sudden increase due to **COVID-19 job losses**, followed by a steady recovery.

        **Forecast Interpretation:**
        - The model predicts **a stable trend** in unemployment.
        - No major increases or decreases are expected in the next few years.

        **Limitations:**
        - This forecast assumes **past trends continue**, but external factors like **recessions, policy changes, or major economic shifts** could change unemployment trends.
        - Further improvements can be made by incorporating **macroeconomic indicators** for better predictions.

        **Conclusion:**  
        - The Texas job market appears **steady**, with unemployment remaining low unless **major economic disruptions occur**.
        """
    )

    # ---- Prophet Forecast (Consumer Confidence) ----
    st.subheader("📊 Prophet Forecast - Consumer Confidence Index")
    df_prophet = df[['consumer_confidence_index_texas']].reset_index().rename(columns={"date": "ds", "consumer_confidence_index_texas": "y"})
    prophet_model = Prophet()
    prophet_model.fit(df_prophet)

    future = prophet_model.make_future_dataframe(periods=forecast_months, freq='M')
    forecast_confidence = prophet_model.predict(future)

    fig2 = prophet_model.plot(forecast_confidence)
    st.pyplot(fig2)

    # ---- Explanation for Consumer Confidence Forecast ----
    st.markdown("### **📈 Understanding Consumer Confidence Trends**")
    st.markdown(
        """
        **What is Consumer Confidence?**  
        - This index reflects **how optimistic or pessimistic consumers feel** about the economy.
        
        **Recent Trends:**
        - Confidence was **high before 2020** but dipped due to the pandemic.
        - Post-2021, fluctuations indicate **economic uncertainty**.

        **Forecast Interpretation:**
        - The model predicts **a potential decline in consumer confidence** over the next few years.
        - Economic factors like **inflation, interest rates, and job growth** can heavily influence this trend.

        **Conclusion:**  
        - **Stable consumer confidence = strong economy.**
        - If confidence drops, **business spending and investments may decline**.
        """
    )

# ---- Footer Section ----
st.markdown("---")
st.markdown("🔗 **LinkedIn:** [Cheran Ratnam](https://www.linkedin.com/in/cheranratnam)")
st.markdown("📄 **Data Source:** [Texas Open Data Portal](https://data.texas.gov/dataset/Key-Economic-Indicators/karz-jr5v)")

