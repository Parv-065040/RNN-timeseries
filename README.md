# 📈 Deep Alpha: Multi-Modal LSTM Stock Forecaster

**Project 4 (Capstone): Application of LSTM Models with Text and Timeseries Datasets in Business Decision-Making.**

Deep Alpha is an interactive, industry-ready financial dashboard deployed via Streamlit. It leverages a multi-modal Long Short-Term Memory (LSTM) deep learning architecture to forecast short-term stock price movements by integrating structured historical market data with unstructured financial news sentiment.

## 🎯 Competency Goals Addressed
* **CG1:** Applied functional and management knowledge to solve a real-world financial forecasting problem.
* **CG2:** Demonstrated critical thinking and data-driven decision making through actionable trading policies.
* **CG3:** Exercised ethical and responsible decision-making (Bias audits, API legitimacy, and explainability).
* **CG6:** Applied data-driven insights and analytical tools (TensorFlow, NLTK, Plotly) to extract business value.

---

## 🚀 Features
* **Multi-Input Deep Learning:** Processes 60-day sequences of historical OHLCV data alongside 60-day sequences of rolling news sentiment.
* **Live API Integration:** Fetches real-time market data using the `yfinance` API.
* **On-the-Fly NLP Pipeline:** Cleans text, removes noise, and scores financial sentiment using the NLTK VADER lexicon.
* **Robust Deployment:** Architecture and weights are decoupled (`.weights.h5`) to bypass Keras 2/Keras 3 cloud server serialization conflicts.
* **Managerial Dashboard:** Features interactive Plotly candlestick charts, automated buy/sell/hold recommendations, and KPI tracking.

---

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **Deep Learning Framework:** TensorFlow / Keras (Functional API)
* **Natural Language Processing:** NLTK (VADER SentimentIntensityAnalyzer), Regex
* **Data Processing:** Pandas, NumPy, Scikit-Learn (MinMaxScaler)
* **Frontend UI/UX:** Streamlit
* **Data Visualization:** Plotly Graph Objects

---

## 📁 Repository Structure
```text
├── app.py                      # Main Streamlit dashboard and inference script
├── requirements.txt            # Explicit dependency list for Streamlit Cloud
├── lstm_weights.weights.h5     # Decoupled LSTM mathematical weights (Keras 2/3 safe)
├── scaler_ts.pkl               # Joblib MinMax scaler for timeseries data
├── scaler_text.pkl             # Joblib MinMax scaler for text sentiment data
└── README.md                   # Project documentation
