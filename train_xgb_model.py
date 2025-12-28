import pandas as pd
import numpy as np
import xgboost as xgb
import pickle

# ------------------ LOAD DATA ------------------
# Use your stock CSV (example: AAPL.csv)
df = pd.read_csv("AAPL.csv")

df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
df.dropna(subset=["Date"], inplace=True)
df.set_index("Date", inplace=True)

df = df[["Adj Close"]].rename(columns={"Adj Close": "Adj_Close"})

# ------------------ FEATURE ENGINEERING ------------------
df["log_return"] = np.log(df["Adj_Close"] / df["Adj_Close"].shift(1))
df.dropna(inplace=True)

def create_features(data, lags=10):
    df_feat = data.copy()
    for lag in range(1, lags + 1):
        df_feat[f"ret_lag_{lag}"] = df_feat["log_return"].shift(lag)
    df_feat["rolling_mean_5"] = df_feat["log_return"].rolling(5).mean()
    df_feat["rolling_std_5"] = df_feat["log_return"].rolling(5).std()
    return df_feat

df_feat = create_features(df).dropna()

X = df_feat.drop(["Adj_Close", "log_return"], axis=1)
y = df_feat["log_return"]

# ------------------ TRAIN XGBOOST ------------------
model = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42
)

model.fit(X, y)

# ------------------ SAVE MODEL ------------------
with open("xgb_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ xgb_model.pkl created successfully")
