# AI-Powered Stock Market Trend Predictor (Terminal-based)
A beginner friendly, AI-powered terminal based application that predicts the next short-term movement (up or down) of a given stock using Logistic Regression, based on real-time stock data fetched from Kaggle. Also generates a trend graph with predicted direction.

# Project Overview
This project uses python, machine learning (logistic regression), and seaborn for visualization to help users:

- Fetch recent stock data for any company
- Analyze the trend using engineered financial indicators
- Predict wheather the stock is likely to go up or down
- Save a clear PNG graph of the trend with predicted movement

Built to run entirely in the terminal, making it ideal for CLI-first environments or data science learners who want to combine AI with financial data.
----------

# Features

- Fetches real-time data using 'yfinance'
- Preprocesses and engineers features like:
  1. Percentage Change ('pct_change')
  2. Moving Average ('MA_5')
  3. Volume

- Trains a Logistic Regression model
- Outputs prediction result ('Uptrend' or 'Downtrend')
- Saves a 'PNG' graph showing trend + prediction using seaborn
- CLI input: Just enter the Company name

# Tech Stack 

- Python 3.13+
- ['yfinance']= stock data
- ['pandas']= data hanndling
- ['numpy']= numeric ops
- ['scikit-learn']= ML model
- ['seaborn']= plotting
- ['matplotlib']= backend for seaborn

# Setup Instructions

'''bash
git clone https://github.com/Adijusting/Stock-Trend-Predictor.git
cd Stock-Trend-Predictor

# Author
Aditya Deshpande
