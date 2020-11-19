import market_data
from keras import models
from normalize import chunck, normalize
from dateutil.parser import parse
from datetime import datetime, timedelta


def evaluate(model, ticker, start_date, end_date, threashold=0.5):
    # model prediction, just hold, perfect performance
    period = model.input_shape[1]

    market = padded_market(ticker, start_date, end_date, period, 1)
    chuncked = normalize(chunck(market[:-1], period))
    predictions = model.predict(chuncked)
    model_buy = [x > threashold for x in predictions]
    just_buy = [True for x in predictions]
    return \
        evaluate_performance(model_buy, market[period-1:]), \
        evaluate_performance(just_buy, market[period-1:]), \
        perfect_performance(len(model_buy), market[period-1:]), \
        worst_performance(len(model_buy), market[period-1:])


def padded_market(ticker, start_date, end_date, leftpad, rightpad):
    begin = parse(start_date)
    end = parse(end_date)
    ybegin = begin - timedelta(days=leftpad * 2 + 7)
    yend = end + timedelta(days=rightpad * 2 + 7)
    market = market_data.download(ticker, str(ybegin.date()), str(yend.date()))
    left = 0
    right = len(market)
    while True:
        timestamp = market.axes[0][left]
        date = datetime(timestamp.year, timestamp.month, timestamp.day)
        if date >= begin:
            break
        left += 1

    while True:
        timestamp = market.axes[0][right-1]
        date = datetime(timestamp.year, timestamp.month, timestamp.day)
        if date <= end:
            break
        right -= 1

    return market[left-leftpad:right+rightpad]


def evaluate_performance(actions, market):
    money = 1
    for i in range(1, len(actions)):
        if actions[i]:
            money *= market.Close[i+1] / market.Close[i]

    return money


def perfect_performance(count, market):
    money = 1
    for i in range(count):
        if market.Close[i + 1] / market.Close[i] > 1:
            money *= market.Close[i+1] / market.Close[i]

    return money


def worst_performance(count, market):
    money = 1
    for i in range(count):
        if market.Close[i + 1] / market.Close[i] < 1:
            money *= market.Close[i + 1] / market.Close[i]

    return money