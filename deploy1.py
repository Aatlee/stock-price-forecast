import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Stock Price Forecast",
    page_icon="📈",
    layout="wide"
)

# ------------------ HEADER ------------------
st.markdown(
    """
    <h1 style='text-align: center;'>📊 Stock Price Forecasting Dashboard</h1>
    <p style='text-align: center; color: grey;'>
    XGBoost-based short-term stock price forecasting
    </p>
    """,
    unsafe_allow_html=True
)

# ------------------ SIDEBAR ------------------
st.sidebar.markdown("## ⚙️ Controls")
st.sidebar.markdown("Upload historical stock data to generate forecasts.")

uploaded_file = st.sidebar.file_uploader(
    "📂 Upload CSV file",
    type=["csv"],
    help="CSV must contain Date and Adj Close columns"
)

forecast_days = st.sidebar.slider(
    "📅 Forecast Horizon (days)",
    min_value=5,
    max_value=60,
    value=30
)

st.sidebar.markdown("---")
st.sidebar.info(
    "📌 **Model**: XGBoost\n\n"
    "📌 **Features**: Lag returns, rolling mean, volatility\n\n"
    "📌 **Target**: Log return"
)

# ------------------ LOAD MODEL (CLOUD SAFE) ------------------
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "xgb_model.pkl")
    if not os.path.exists(model_path):
        st.error("❌ Model file 'xgb_model.pkl' not found in repository.")
        st.stop()
    with open(model_path, "rb") as f:
        return pickle.load(f)

model = load_model()

# ------------------ NO FILE GUARD ------------------
if uploaded_file is None:
    st.warning("⬅️ Please upload a CSV file from the sidebar to continue.")
    st.stop()

# ------------------ DATA LOADING ------------------
df = pd.read_csv(uploaded_file)

# Safe date parsing
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
df.dropna(subset=["Date"], inplace=True)

df.set_index("Date", inplace=True)
df = df[["Adj Close"]].rename(columns={"Adj Close": "Adj_Close"})

# ------------------ RETURNS ------------------
df["log_return"] = np.log(df["Adj_Close"] / df["Adj_Close"].shift(1))
df.dropna(inplace=True)

# ------------------ FEATURE ENGINEERING ------------------
def create_features(data, lags=10):
    df_feat = data.copy()
    for lag in range(1, lags + 1):
        df_feat[f"ret_lag_{lag}"] = df_feat["log_return"].shift(lag)

    df_feat["rolling_mean_5"] = df_feat["log_return"].rolling(5).mean()
    df_feat["rolling_std_5"] = df_feat["log_return"].rolling(5).std()
    return df_feat

df_feat = create_features(df).dropna()

# ------------------ FORECASTING ------------------
last_row = df_feat.iloc[-1:].copy()
last_price = df["Adj_Close"].iloc[-1]
future_prices = []

for _ in range(forecast_days):
    X_last = last_row.drop(["Adj_Close", "log_return"], axis=1)
    pred_return = model.predict(X_last)[0]

    last_price *= np.exp(pred_return)
    future_prices.append(last_price)

    for lag in range(10, 1, -1):
        last_row[f"ret_lag_{lag}"] = last_row[f"ret_lag_{lag - 1}"]

    last_row["ret_lag_1"] = pred_return

    recent_returns = last_row[[f"ret_lag_{i}" for i in range(1, 6)]].values.flatten()
    last_row["rolling_mean_5"] = recent_returns.mean()
    last_row["rolling_std_5"] = recent_returns.std()
    last_row["log_return"] = pred_return

# ------------------ FUTURE DATES ------------------
future_dates = pd.bdate_range(
    start=df.index[-1],
    periods=forecast_days + 1
)[1:]

forecast_df = pd.DataFrame(
    {"Forecast_Price": future_prices},
    index=future_dates
)

# ------------------ METRICS ------------------
st.markdown("## 📌 Key Metrics")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Last Closing Price",
    f"{df['Adj_Close'].iloc[-1]:.2f}"
)

col2.metric(
    "Final Forecast Price",
    f"{forecast_df['Forecast_Price'].iloc[-1]:.2f}"
)

pct_change = (
    (forecast_df["Forecast_Price"].iloc[-1] - df["Adj_Close"].iloc[-1])
    / df["Adj_Close"].iloc[-1] * 100
)

col3.metric(
    "Expected Change",
    f"{pct_change:.2f} %",
    delta=f"{pct_change:.2f} %"
)

# ------------------ TABS ------------------
tab1, tab2 = st.tabs(["📈 Forecast Chart", "📋 Forecast Table"])

# ------------------ PLOT ------------------
with tab1:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df["Adj_Close"].iloc[-120:], label="Historical Price")
    ax.plot(
        forecast_df.index,
        forecast_df["Forecast_Price"],
        linestyle="--",
        label="Forecast"
    )
    ax.set_title("Stock Price Forecast", fontsize=16)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)

# ------------------ TABLE ------------------
with tab2:
    st.dataframe(
        forecast_df.style.format("{:.2f}"),
        use_container_width=True
    )

# ------------------ FOOTER ------------------
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: grey;'>"
    "Built with Streamlit • XGBoost • Financial Time Series"
    "</p>",
    unsafe_allow_html=True
)
