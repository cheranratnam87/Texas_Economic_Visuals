import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
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
tab1, tab2, tab3 = st.tabs(["📈 Data Exploration", "📊 Forecasts", "🏛️ USA vs Texas Comparisons"])

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
    default_state = 'unemployment_tx' if 'unemployment_tx' in state_columns else state_columns[0]
    selected_state = st.selectbox("Select State Indicator", state_columns, index=state_columns.index(default_state))

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
    st.subheader("👨‍💼 ARIMA Forecast - Unemployment Rate (TX)")

    @st.cache_data
    def fit_arima_model(data):
        model = ARIMA(data, order=(1,1,1)).fit()
        return model

    arima_model = fit_arima_model(df['unemployment_tx'])
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

    # Define Texas colors
    TEXAS_RED = "#BF0D3E"
    TEXAS_BLUE = "#002147"
    TEXAS_WHITE = "#FFFFFF"

    # ---- 📅 SARIMA (Seasonal ARIMA) Forecast ----
    st.subheader(f"📉 SARIMA (Seasonal ARIMA) Forecast for {selected_state.replace('_', ' ').title()}")

    @st.cache_data
    def fit_sarima_model(data):
        model = SARIMAX(data, order=(1,1,1), seasonal_order=(1,1,1,12)).fit()
        return model

    sarima_model = fit_sarima_model(df[selected_state])
    forecast_sarima = sarima_model.forecast(steps=forecast_months)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df.index, df[selected_state], label="Actual", color=TEXAS_BLUE)
    ax.plot(pd.date_range(df.index[-1], periods=forecast_months, freq='ME'), forecast_sarima, linestyle="dashed", color=TEXAS_RED, label="SARIMA Forecast")
    ax.legend()
    st.pyplot(fig)

    st.markdown("### 🔍 Key Insights")
    st.markdown(
    """
    - **SARIMA captures both trend & seasonality**, making it ideal for unemployment cycles.
    - **Unlike ARIMA, it considers annual/seasonal patterns** (e.g., COVID or economic recessions).
    - **Next Steps:** Compare SARIMA vs. ARIMA and test longer forecast horizons.
    """
    )

    # ---- Holt-Winters (Exponential Smoothing) Forecast ----
    st.subheader(f"📊 Holt-Winters (Exponential Smoothing) Forecast for {selected_state.replace('_', ' ').title()}")

    @st.cache_data
    def fit_ets_model(data):
        seasonal_period = min(6, max(2, len(data) // 2))  # Adjust seasonal periods dynamically
        model = ExponentialSmoothing(data, trend="add", seasonal="add", seasonal_periods=seasonal_period).fit()
        return model

    ets_input = df[selected_state].replace(0, 0.01).dropna()
    if len(ets_input) >= 12:
        ets_model = fit_ets_model(ets_input)
        forecast_ets = ets_model.forecast(forecast_months)
    else:
        st.warning("⚠ Not enough data for Holt-Winters model. Using simple moving average instead.")
        forecast_ets = ets_input.rolling(window=3, min_periods=1).mean().iloc[-1]
        forecast_ets = [forecast_ets] * forecast_months  # Repeat last value

    # Plot ETS Forecast
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df.index, df[selected_state], label="Actual", color=TEXAS_BLUE)
    ax.plot(pd.date_range(df.index[-1], periods=forecast_months, freq='ME'), forecast_ets, linestyle="dashed", color=TEXAS_RED, label="ETS Forecast")
    ax.legend()
    st.pyplot(fig)

    st.markdown("### 🔍 Insights on ETS Model")
    st.markdown(
    """
    - **Holt-Winters captures both trends & seasonality**, making it useful for unemployment cycles.
    - **More flexible than ARIMA for short-term forecasting.**
    - **If there isn’t enough data, a simple trend model is used instead.**
    """
    )

    # ---- 💳 Consumer Confidence Index Forecast ----
    st.header("💰 Consumer Confidence Index Forecast")

    df_confidence = df[['consumer_confidence_index_texas']].reset_index().rename(columns={"date": "ds", "consumer_confidence_index_texas": "y"})

    @st.cache_data
    def fit_prophet_model(data):
        model = Prophet()
        model.fit(data)
        return model

    prophet_model = fit_prophet_model(df_confidence)
    future = prophet_model.make_future_dataframe(periods=forecast_months, freq='ME')
    forecast_prophet = prophet_model.predict(future)

    arima_confidence = fit_arima_model(df['consumer_confidence_index_texas'])
    forecast_arima_confidence = arima_confidence.forecast(steps=forecast_months)

    sarima_confidence = fit_sarima_model(df['consumer_confidence_index_texas'])
    forecast_sarima_confidence = sarima_confidence.forecast(steps=forecast_months)

    ets_confidence = fit_ets_model(df['consumer_confidence_index_texas'].dropna())
    forecast_ets_confidence = ets_confidence.forecast(steps=forecast_months)

    # ---- Plot all forecasts ----
    fig, ax = plt.subplots(figsize=(10, 5))

    # Actual Data
    ax.plot(df.index, df['consumer_confidence_index_texas'], label="Actual", color=TEXAS_BLUE)

    # Forecasts
    ax.plot(pd.date_range(df.index[-1], periods=forecast_months, freq='ME'), forecast_prophet['yhat'][-forecast_months:], linestyle="dashed", color="green", label="Prophet Forecast")
    ax.plot(pd.date_range(df.index[-1], periods=forecast_months, freq='ME'), forecast_arima_confidence, linestyle="dashed", color="orange", label="ARIMA Forecast")
    ax.plot(pd.date_range(df.index[-1], periods=forecast_months, freq='ME'), forecast_sarima_confidence, linestyle="dashed", color="red", label="SARIMA Forecast")
    ax.plot(pd.date_range(df.index[-1], periods=forecast_months, freq='ME'), forecast_ets_confidence, linestyle="dashed", color="purple", label="ETS Forecast")

    ax.legend()
    st.pyplot(fig)

    # ---- Key Insights ----
    st.markdown("### 🔍 Key Insights")
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

    st.markdown(
    """
    - **Prophet:** Captures trends & seasonality, robust to missing data.
    - **ARIMA:** Works well if data is non-seasonal and follows a linear trend.
    - **SARIMA:** Best for seasonal patterns, considers yearly trends.
    - **Holt-Winters (ETS):** Smooths trends and is useful for short-term forecasting.

    **Next Steps:**
    - Compare model accuracy and refine hyperparameters.
    - Monitor external factors like interest rates & inflation that impact consumer confidence.
    """
    )

# ---- TAB 3: USA vs Texas Comparisons ----
with tab3:
    st.session_state["active_tab"] = "USA vs Texas Comparisons"
    st.header("🏛️ USA vs 🤠 Texas Economic Comparison")

    # 👷 Unemployment Comparison
    st.subheader("📆 Unemployment Rate: Texas vs USA")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(df.index, df['unemployment_tx'], label="Texas Unemployment", color=TEXAS_BLUE, linestyle="-")
    ax1.plot(df.index, df['unemployment_us'], label="USA Unemployment", color="red", linestyle="--")
    ax1.set_title("Unemployment Rate Comparison (TX vs USA)")
    ax1.legend()
    st.pyplot(fig1)

    # 🔍 Insights
    st.markdown("### 💡 Key Takeaways on Unemployment")
    st.markdown(
        """
        - **Texas unemployment rates** have historically been **lower** than the **USA average**, indicating a strong job market.
        - **Economic shocks (e.g., 2008 Recession, COVID-19)** impacted both, but Texas often recovered faster.
        - **Industries in Texas (energy, tech, manufacturing)** provide **economic resilience** compared to the national average.
        """
    )

    # 💰 Consumer Confidence Comparison
    st.subheader("💳 Consumer Confidence Index: Texas vs USA")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(df.index, df['consumer_confidence_index_texas'], label="Texas Confidence Index", color=TEXAS_BLUE, linestyle="-")
    ax2.plot(df.index, df['consumer_confidence_index_us'], label="USA Confidence Index", color="red", linestyle="--")
    ax2.set_title("Consumer Confidence Index Comparison (TX vs USA)")
    ax2.legend()
    st.pyplot(fig2)

    # 🔍 Insights
    st.markdown("### 🔍 Key Takeaways on Consumer Confidence")
    st.markdown(
        """
        - **Texas consumers are generally more optimistic** than the national average, reflecting a strong local economy.
        - **Post-2020 trends show more volatility** due to supply chain disruptions, inflation, and interest rate shifts.
        - **A higher consumer confidence index** means more consumer spending, which boosts business activity in Texas.
        """
    )

# ---- Footer Section ----
st.markdown("---")
st.markdown("🔗 **LinkedIn:** [Cheran Ratnam](https://www.linkedin.com/in/cheranratnam)")
st.markdown("📄 **Data Source:** [Texas Open Data Portal](https://data.texas.gov/dataset/Key-Economic-Indicators/karz-jr5v)")
