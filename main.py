import joblib
import warnings
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import numpy as np
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def log(msg):
    print(msg)
    sys.stdout.flush()
"""
Terminal-based stock trend predictor
"""

model = joblib.load("C:/Users/Aditya/OneDrive/Desktop/Python_projs/Stock_Predictor/models/LogisticRegression.pkl")
scaler = joblib.load("C:/Users/Aditya/OneDrive/Desktop/Python_projs/Stock_Predictor/models/scaler.pkl")

log("Model and scaler loaded successfully")

symbol = input("Enter stock ticker symbol (e.g, BPCL, CIPLA): ").upper()
if not symbol.endswith(".NS") and not symbol.endswith(".BO"):
    symbol+=".NS"

data = yf.download(symbol, period = "7d", group_by="ticker", auto_adjust=False, progress=False)

log("Raw data preview: ")
log(data.tail())
log(f"Data length: {len(data)} rows")
data.dropna(inplace=True)
    
if len(data) < 6:
    log("Not enough data to analyze, try another symbol")
    exit()
        
if isinstance(data.columns, pd.MultiIndex):
    data.columns=['_'.join(col).strip() for col in data.columns.values]

data.fillna(data.mean(numeric_only=True), inplace=True)
    
# print("\nColumns: ", data.columns.tolist())
# Taking user input
try:
    close_col = f"{symbol}_Close"
    volume_col = f"{symbol}_Volume"
        # Calculate features
    # log("\nFeature enginering step: ")
    close_prices = data[close_col]
    # log("Close prices extracted")
    pct_change = ((close_prices.iloc[-1] - close_prices.iloc[-2]) / close_prices.iloc[-2])*100
    ma_5 = close_prices.iloc[-6:-1].mean()
    volume = data[volume_col].iloc[-1]
    # log("Features calculated")
        
    log("\nExtracted features: ")
    log(f"% Change: {pct_change:.2f}")
    log(f"MA_5: {ma_5:.2f}")
    log(f"Volume: {int(volume)}")
    
    warnings.filterwarnings("ignore", category=UserWarning)
        
        # Prepare input for model
    features = np.array([[pct_change, ma_5, volume]])
    # log("\nFeatures array created")
    scaled_input = scaler.transform(features)
    # log("Features scaled")
        
        # Predict
    prediction = model.predict(scaled_input)[0]
    confidence= model.predict_proba(scaled_input)[0][prediction]*100
    label = "Uptred expected" if prediction ==1 else "Downtrend expected"
        
    log(f"\nPrediction result: {label}")
    log(f"\nConfidence: {confidence:.2f}%")
    
    plot_df = pd.DataFrame({
        "Close": close_prices,
        "MA_5": close_prices.rolling(window=5).mean()
    })
    
    plot_df["Date"]=plot_df.index
    
    # Set seaborn style
    sns.set(style="whitegrid")
    
    # Create lineplot
    plt.figure(figsize=(10,6))
    sns.lineplot(x="Date", y="Close", data=plot_df, label="Close Price", marker="o", color="blue")
    sns.lineplot(x="Date", y="MA_5", data=plot_df, label="5-day moving average", linestyle="--", color="orange")
    
    # Mark prediciton point
    last_day = close_prices.index[-1]
    last_price = close_prices.iloc[-1]
    color="green" if prediction==1 else "red"
    plt.scatter(last_day, last_price, s=100, color=color, label="AI Prediction")
    
    # Titles and labels
    plt.title(f"{symbol} Stock Trend with AI prediction", fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("Price (INR)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    
    # Save 
    plt.savefig(f"{symbol}_trend_seaborn.png")
    print(f"Grpah saved as {symbol}_trend_seaborn.png")
        
except Exception as e:
    log(f"Error fetching data for {symbol}: {e}")
    print(f"Availabel columns: {data.columns.tolist()}")
    exit()
