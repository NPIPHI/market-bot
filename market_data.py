import yfinance as yf
import pandas as pd
import os
import ta
from sklearn.preprocessing import MinMaxScaler, RobustScaler


def download(ticker):
    cached_files = os.listdir("data")
    if ticker + ".csv" in cached_files:
        return pd.read_csv("data/" + ticker + ".csv", header=0, index_col="Date", parse_dates=True)
    else:
        return None
        # yf api is returning incorrect results
        # data = yf.Ticker(ticker).history(period="max")
        # data.to_csv('data/' + ticker + ".csv")
        # return data


def get_data(ticker, start_date, end_date):
    return download(ticker)[start_date: end_date]


def get_ta_data_raw(ticker, start, end):
    data = get_data(ticker, start, end)
    data.dropna(inplace=True)
    data = ta.add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume",
                                  fillna=True)
    data.drop(["Open", "High", "Low", "Volume", "Adj Close", "Volume"], axis=1, inplace=True)
    return data


def get_ta_data(ticker, start, end):
    data = get_data(ticker, start, end)
    data.dropna(inplace=True)
    data = ta.add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume",
                                  fillna=True)
    data.drop(["Open", "High", "Low", "Volume", "Adj Close", "Volume"], axis=1, inplace=True)
    close_scalar = RobustScaler()
    close_scalar.fit(data[['Close']])
    scalar = RobustScaler()
    return pd.DataFrame(scalar.fit_transform(data), columns=data.columns, index=data.index), close_scalar, scalar