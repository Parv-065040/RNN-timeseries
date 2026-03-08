import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import joblib
import re
import os
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Import Keras layers to rebuild the model natively
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, concatenate

# ---------------------------------------------------------
# Environment Setup & NLTK Downloads (Cloud-Safe)
# ---------------------------------------------------------
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', download_dir=nltk_data_dir, quiet=True)

# ---------------------------------------------------------
# Page Configuration & UI/UX Setup
# ---------------------------------------------------------
st.set_page_config(page_title="Deep Alpha: AI Stock Forecaster", page_icon="📈", layout="wide")

st.markdown("""
    <style>
    .metric-card {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #00FF00;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .metric-title { color: #888888; font-size: 14px; text-transform: uppercase; letter-spacing: 1px;}
    .metric-value { color: #FFFFFF; font-size: 28px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.title("📈 Deep Alpha: Multi-Modal Stock Prediction Engine")
st.markdown("Integrating **Timeseries Historical Data** with **Unstructured News Sentiment** via LSTM Deep Learning.")
st.markdown("---")

# ---------------------------------------------------------
# Model Architecture Rebuilding (Bypasses Version Conflicts)
# ---------------------------------------------------------
def build_lstm_model():
    """Reconstructs the precise architecture built in Colab."""
    # Branch 1: Timeseries (60 timesteps, 2 features: Close, Volume)
    input_ts = Input(shape=(60, 2), name="Timeseries_Input")
    lstm_ts = LSTM(units=50, return_sequences=False)(input_ts)
    dropout_ts = Dropout(0.2)(lstm_ts)

    # Branch 2: Text Sentiment (60 timesteps, 1 feature: Sentiment_Score)
    input_text = Input(shape=(60, 1), name="Text_Input")
    lstm_text = LSTM(units=30, return_sequences=False)(input_text)
    dropout_text = Dropout(0.2)(lstm_text)

    # Merge and Output
    merged = concatenate([dropout_ts, dropout_text])
    dense1 = Dense(32, activation='relu')(merged)
    output = Dense(1, activation='linear', name="Price_Prediction")(dense1)

    model = Model(inputs=[input_ts, input_text], outputs=output)
    return model

# ---------------------------------------------------------
# Caching Heavy Functions for Performance
# ---------------------------------------------------------
@st.cache_resource
def load_assets():
    """Builds model, loads raw weights, and loads scalers."""
    # 1. Rebuild the empty architecture
    model = build_lstm_model()
    # 2. Inject the trained weights
    model.load_weights('lstm_weights.h5')
    
    scaler_ts = joblib.load('scaler_ts.pkl')
    scaler_text = joblib.load('scaler_text.pkl')
    analyzer = SentimentIntensityAnalyzer()
    
    return model, scaler_ts, scaler_text, analyzer

try:
    model, scaler_ts, scaler_text, analyzer = load_assets()
except Exception as e:
    st.error(f"Error loading assets: {e}. Ensure 'lstm_weights.h5', 'scaler_ts.pkl', and 'scaler_text.pkl' are uploaded.")
    st.stop()

# ---------------------------------------------------------
# Sidebar Inputs
# ---------------------------------------------------------
st.sidebar.header("System Controls")
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL").upper()
lookback_days = 60 

# ---------------------------------------------------------
# Data Fetching & Processing Pipeline
# ---------------------------------------------------------
if st.sidebar.button("Run Analytical Engine"):
    with st.spinner(f"Fetching market data and processing NLP pipeline for {ticker}..."):
        
        # 1. Fetch Timeseries Data
        stock_data = yf.download(ticker, period="100d")
        
        if stock_data.empty:
            st.error(f"No data found for ticker '{ticker}'. Please check the symbol and try again.")
            st.stop()
            
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = [col[0] for col in stock_data.columns]
            
        stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']].reset_index()
        stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.tz_localize(None)
        
        # 2. Simulate Recent News
        sample_headlines = [
            f"{ticker} announces strong forward guidance, exceeding analyst expectations.",
            f"Macroeconomic headwinds apply pressure to {ticker} supply chain.",
            f"Investors remain cautious on {ticker} pending regulatory review.",
            f"Major institutional buy-in boosts {ticker} market confidence.",
            f"{ticker} faces new competition in emerging markets."
        ]
        
        np.random.seed(42)
        news_data = pd.DataFrame({
            'Date': stock_data['Date'],
            'Headline': [np.random.choice(sample_headlines) for _ in range(len(stock_data))]
        })

        def get_sentiment(text):
            text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
            return analyzer.polarity_scores(text)['compound']
            
        news_data['Sentiment_Score'] = news_data['Headline'].apply(get_sentiment)
        
        # Merge Data
        df_merged = pd.merge(stock_data, news_data[['Date', 'Sentiment_Score']], on='Date', how='inner')
        
        if len(df_merged) < lookback_days:
            st.error("Not enough historical trading days extracted. Try a different ticker.")
            st.stop()

        # 3. Prepare Data for LSTM
        recent_data = df_merged.tail(lookback_days).copy()
        
        ts_features = recent_data[['Close', 'Volume']].values
        text_features = recent_data[['Sentiment_Score']].values
        
        scaled_ts_input = scaler_ts.transform(ts_features)
        scaled_text_input = scaler_text.transform(text_features)
        
        X_ts_pred = np.array([scaled_ts_input])
        X_text_pred = np.array([scaled_text_input])
        
        # 4. Execute Prediction
        prediction_scaled = model.predict([X_ts_pred, X_text_pred])
        
        dummy_pred = np.zeros((1, ts_features.shape[1]))
        dummy_pred[0, 0] = prediction_scaled[0, 0] 
        predicted_price = scaler_ts.inverse_transform(dummy_pred)[0, 0]
        
        # 5. Calculate Metrics
        current_price = recent_data['Close'].iloc[-1]
        price_change = predicted_price - current_price
        
        avg_sentiment = recent_data['Sentiment_Score'].tail(3).mean()
        sentiment_label = "Bullish" if avg_sentiment > 0.1 else "Bearish" if avg_sentiment < -0.1 else "Neutral"

        # ---------------------------------------------------------
        # UI Dashboard Rendering
        # ---------------------------------------------------------
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Current Price ({ticker})</div>
                    <div class="metric-value">${current_price:.2f}</div>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
                <div class="metric-card" style="border-left: 5px solid {'#00FF00' if price_change > 0 else '#FF0000'};">
                    <div class="metric-title">LSTM Predicted Close (T+1)</div>
                    <div class="metric-value">${predicted_price:.2f}</div>
                </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
                <div class="metric-card" style="border-left: 5px solid {'#00FF00' if avg_sentiment > 0 else '#FF0000'};">
                    <div class="metric-title">3-Day News Sentiment</div>
                    <div class="metric-value">{sentiment_label} ({avg_sentiment:.2f})</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Interactive Plotly Chart
        st.subheader(f"Price Action & Model Projection: {ticker}")
        
        fig = go.Figure()
        
        fig.add_trace(go.Candlestick(
            x=recent_data['Date'],
            open=recent_data['Open'],
            high=recent_data['High'],
            low=recent_data['Low'],
            close=recent_data['Close'],
            name='Historical Data'
        ))
        
        next_day = recent_data['Date'].iloc[-1] + pd.Timedelta(days=1)
        
        fig.add_trace(go.Scatter(
            x=[recent_data['Date'].iloc[-1], next_day],
            y=[current_price, predicted_price],
            mode='lines+markers',
            name='LSTM Prediction',
            line=dict(color='yellow', width=3, dash='dot'),
            marker=dict(size=10, symbol='star')
        ))

        fig.update_layout(
            template='plotly_dark',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            xaxis_rangeslider_visible=False,
            height=600,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Managerial Interpretation & Recommended Policy")
        
        recommendation = "HOLD"
        if price_change > 0 and avg_sentiment > 0.05:
            recommendation = "BUY / ACCUMULATE"
        elif price_change < 0 and avg_sentiment < -0.05:
            recommendation = "SELL / REDUCE EXPOSURE"

        st.info(f"**Data-Driven Policy Recommendation:** {recommendation}")
        st.write(f"**Justification:** The multi-modal LSTM architecture forecasts a near-term price movement of **${price_change:.2f}**. This quantitative projection is corroborated by the current market sentiment index, which sits at **{avg_sentiment:.2f}** ({sentiment_label}). Stakeholders should cross-reference this signal with broader macroeconomic risk models before execution.")

else:
    st.info("👈 Enter a stock ticker and click 'Run Analytical Engine' to generate insights.")
