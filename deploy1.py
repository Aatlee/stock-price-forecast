import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="Stock Price Forecast",
    page_icon="📈",
    layout="wide"
)

# ================== HEADER ==================
st.markdown(
    """
    <h1 style='text-align: center;'>📊 Stock Price Forecasting Dashboard</h1>
    <p style='text-align: center; color: grey;'>
    XGBoost-based short-term stock price forecasting
    </p>
    """,
    unsafe_allow_html=True
)

# ================== SIDEBAR ==================
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

# ================== LOAD MODEL (CLOUD SAFE) ==================
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "xgb_model.pkl")

    if not os.path.exists(model_path):
        st.error("❌ xgb_model.pkl not found. Upload it to GitHub and redeploy.")
        st.stop()

    with open(model_path, "rb") as f:
        return pickle.load(f)

model = load_model()

# ================== NO FILE GUARD ==================
if uploaded_file is None:
    st.warning("⬅️ Please upload a CSV file from the sidebar to continue.")
    st.stop()

# ================== DATA LOADING ==================
df = pd.read_csv(uploaded_file)

df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
df.dropna(subset=["Date"], inplace=True)

df.set_index("Date", inplace=True)
df = df[["Adj Close"]].rename(columns={"Adj Close": "Adj_Close"})

# ================== RETURNS ==================
df["log_return"] = np.log(df["Adj_Close"] / df["Adj_Close"].shift(1))
df.dropna(inplace=True)

# ================== FEATURE ENGINEERING ==================
def create_features(data, lags=10):
    feat = data.copy()
    for lag in range(1, lags + 1):
        feat[f"ret_lag_{lag}"] = feat["log_return"].shift(lag)

    feat["rolling_mean_5"] = feat["log_return"].rolling(5).mean()
    feat["rolling_std_5"] = feat["log_return"].rolling(5).std()
    return feat

df_feat = create_features(df).dropna()

# ================== MODEL ACCURACY (XGBOOST) ==================
split = int(len(df_feat) * 0.8)

X_all = df_feat.drop(["Adj_Close", "log_return"], axis=1)
y_all = df_feat["log_return"]

X_test = X_all.iloc[split:]
y_test = y_all.iloc[split:]

y_pred = model.predict(X_test)

rmse_ret = np.sqrt(mean_squared_error(y_test, y_pred))
mae_ret = mean_absolute_error(y_test, y_pred)

actual_price = df_feat["Adj_Close"].iloc[-len(y_test):].values
pred_price = actual_price[0] * np.exp(np.cumsum(y_pred))

rmse_price = np.sqrt(mean_squared_error(actual_price, pred_price))
mae_price = mean_absolute_error(actual_price, pred_price)

st.markdown("## 🎯 Model Accuracy (XGBoost)")

a1, a2, a3, a4 = st.columns(4)
a1.metric("RMSE (Returns)", f"{rmse_ret:.6f}")
a2.metric("MAE (Returns)", f"{mae_ret:.6f}")
a3.metric("RMSE (Price)", f"{rmse_price:.2f}")
a4.metric("MAE (Price)", f"{mae_price:.2f}")

# ================== FORECASTING ==================
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

    recent = last_row[[f"ret_lag_{i}" for i in range(1, 6)]].values.flatten()
    last_row["rolling_mean_5"] = recent.mean()
    last_row["rolling_std_5"] = recent.std()
    last_row["log_return"] = pred_return

# ================== FUTURE DATES ==================
future_dates = pd.bdate_range(
    start=df.index[-1],
    periods=forecast_days + 1
)[1:]

USD_TO_INR = 83

forecast_df = pd.DataFrame(
    {
        "Forecast_Price_USD": future_prices,
        "Forecast_Price_INR": np.array(future_prices) * USD_TO_INR
    },
    index=future_dates
)

# ================== METRICS ==================
st.markdown("## 📌 Key Metrics")

c1, c2, c3 = st.columns(3)

c1.metric(
    "Last Closing Price (USD)",
    f"{df['Adj_Close'].iloc[-1]:.2f}"
)

c2.metric(
    "Final Forecast Price (USD)",
    f"{forecast_df['Forecast_Price_USD'].iloc[-1]:.2f}"
)

pct_change = (
    (forecast_df["Forecast_Price_USD"].iloc[-1] - df["Adj_Close"].iloc[-1])
    / df["Adj_Close"].iloc[-1] * 100
)

c3.metric(
    "Expected Change",
    f"{pct_change:.2f} %",
    delta=f"{pct_change:.2f} %"
)

# ================== TABS ==================
tab1, tab2 = st.tabs(["📈 Forecast Chart", "📋 Forecast Table"])

# ================== PLOT ==================
with tab1:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df["Adj_Close"].iloc[-120:], label="Historical Price")
    ax.plot(
        forecast_df.index,
        forecast_df["Forecast_Price_USD"],
        linestyle="--",
        label="Forecast (USD)"
    )
    ax.set_title("Stock Price Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig)

# ================== TABLE ==================
with tab2:
    st.dataframe(
        forecast_df.style.format("{:.2f}"),
        use_container_width=True
    )

# ================== FOOTER ==================
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: grey;'>"
    "Built with Streamlit • XGBoost • Financial Time Series"
    "</p>",
    unsafe_allow_html=True
)
