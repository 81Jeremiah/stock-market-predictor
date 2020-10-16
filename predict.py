import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

stocks = pd.read_csv("sphist.csv")

stocks['Date'] = pd.to_datetime(stocks['Date'])

stocks.sort_values(by=['Date'], ascending=True, inplace=True )
stocks.sort_values(by="Date", ascending=True, inplace=True)
stocks.reset_index(drop=True, inplace=True)

# Create indicators
stocks["avg_5"] = stocks["Close"].rolling(5).mean().shift(1)
stocks["avg_30"] = stocks["Close"].rolling(30).mean().shift(1)
stocks["avg_365"] = stocks["Close"].rolling(365).mean().shift(1)

stocks["std_5"] = stocks["Close"].rolling(5).std().shift(1)
stocks["std_365"] = stocks["Close"].rolling(365).std().shift(1)

stocks["avg_5/avg_365"] = stocks["avg_5"]/stocks["avg_365"]
stocks["std_5/std_365"] = stocks["std_5"]/stocks["std_365"]


stocks.dropna(axis=0, inplace=True)

# Create train and test dataframes
train = stocks[stocks["Date"] < datetime(year=2013, month=1, day=1)]
test = stocks[stocks["Date"] >= datetime(year=2013, month=1, day=1)]

# Make predictions with LinearRegression
lr = LinearRegression()
lr.fit(train[["avg_5", "avg_30", "avg_365", "std_5", "std_365", "avg_5/avg_365", "std_5/std_365"]], train["Close"])
predictions = lr.predict(test[["avg_5", "avg_30", "avg_365", "std_5", "std_365", "avg_5/avg_365", "std_5/std_365"]])

# Calculate error metrics
mae = mean_absolute_error(test["Close"], predictions)
mse = mean_squared_error(test["Close"], predictions)
print("MAE: ", mae)
print("MSE: ", mse)
print(stocks.head(15))
print(stocks[stocks["Date"] == datetime(year=1951, month=1, day=2)].index)
    
