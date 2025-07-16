import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

"""
First we will concat all the datasets into one file
"""

# Path to the folder with all CSV files
data_folder = "C:/Users/Aditya/OneDrive/Desktop/Python_projs/Stock_Predictor/data"
files = [f for f in os.listdir(data_folder) if f.endswith(".csv")]

# Loading and combining all datasets
dataframes =[]

for file in files:
    print(f"Checking file: {file}")
    
    try:
        df=pd.read_csv(os.path.join(data_folder,file))
        # print(df.head(3))
        df.rename(columns=lambda x:x.strip().title(), inplace=True)
        
        if "Date" not in df.columns:
            print("Date column missing")
            continue
        
        df["Symbol"]= file.replace(".csv","")
        dataframes.append(df)
        print("File added to list")
        
    except Exception as e:
        print(f"error reading {file}: {e}")
    
# Combining all stock data into one dataframe
all_data = pd.concat(dataframes, ignore_index=True)

# Calculate features for each stock seperately
processed =[]

for symbol in all_data["Symbol"].unique():
    df = all_data[all_data["Symbol"]==symbol].copy()
    
    # Feature Engineering
    df["Pct_Change"]= df["Close"].pct_change()*100
    df["MA_5"] = df["Close"].rolling(window=5).mean()
    df["Target"] = np.where(df["Close"].shift(-1)>df["Close"],1,0)
    
    df = df.dropna()
    processed.append(df)
    
# Final combines dataset
final_df = pd.concat(processed, ignore_index=True)

x=final_df[["Pct_Change","MA_5","Volume"]]
y= final_df["Target"]

# os.makedirs("data", exist_ok=True)
# output_path = "C:/Users/Aditya/OneDrive/Desktop/Python_projs/Stock_Predictor/data/combined_stock_data.csv"
# final_df.to_csv(output_path, index=False)
# print(f"Combined dataset saved successfully to: {output_path}")

"""
Scaling and Training ML models
"""

# Split into training and testing data
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.2, random_state=42)

# Normalizing features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scled = scaler.transform(x_test)

# Training multiple models
model=LogisticRegression()
model.fit(x_train_scaled, y_train)
print("Trained Logistic Regression")

# Saving all models and scaler
os.makedirs("models", exist_ok=True)

joblib.dump(model, "C:/Users/Aditya/OneDrive/Desktop/Python_projs/Stock_Predictor/models/LogisticRegression.pkl")

# save the scalar 
joblib.dump(scaler, "C:/Users/Aditya/OneDrive/Desktop/Python_projs/Stock_Predictor/models/scaler.pkl")
print("All models and scaler saved siccessfully")

