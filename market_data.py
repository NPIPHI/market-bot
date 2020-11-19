import yfinance as yf
import pandas as pd
import os


def get_data(ticker):
    cached_files = os.listdir("tickerCache")
    if ticker in cached_files:
        return pd.read_csv("tickerCache/" + ticker, header=0, index_col="Date", parse_dates=True)
    else:
        data = yf.Ticker(ticker).history(period="max")
        data.to_csv('tickerCache/' + ticker)
        return data


def download(ticker, start_date, end_date):
    return get_data(ticker)[start_date: end_date]