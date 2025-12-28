# 📈 Stock Price Forecasting using XGBoost & Streamlit

An end-to-end **machine learning–based stock price forecasting application** built using **XGBoost** and deployed with **Streamlit**.  
The app performs **time-series feature engineering** on historical stock prices and generates **short-term future price forecasts** through an interactive web dashboard.

---

## 🔍 Project Overview

Stock price prediction is a challenging time-series problem influenced by market volatility and trends.  
This project focuses on **short-term forecasting** by transforming historical prices into informative features such as:

- Log returns  
- Lagged returns  
- Rolling mean  
- Rolling volatility  

A trained **XGBoost regression model** predicts future log returns, which are then converted back into forecasted stock prices.

---

## 🚀 Features

- 📂 Upload historical stock CSV files  
- 📅 Adjustable forecast horizon (5–60 days)  
- 📊 Interactive historical vs forecast price chart  
- 📋 Forecast results in tabular format  
- 📌 Key metrics (last price, final forecast, expected change)

---

## 🛠️ Tech Stack

- **Programming Language:** Python  
- **Machine Learning:** XGBoost  
- **Data Processing:** Pandas, NumPy  
- **Visualization:** Matplotlib  
- **Web App Framework:** Streamlit  

---

## 📁 Project Structure

