from keras import models
from evaluate import evaluate
from train import train_model
from datetime import datetime, timedelta

# model = train_model("GOOG", "2010-1-1", "2018-1-1", 64)
# model = models.load_model("models/test")
# print(evaluate(model, "GOOG", "2018-1-2", "2018-1-2"))

model = train_model("SPY", "2000-1-1", "2020-4-1", 64)
models.save_model(model, "models/test")
# model = models.load_model("models/test")
print(evaluate(model, "SPY", "2000-1-1", "2020-4-1"))

# date = datetime(2010, 1, 1)
# money = 1
# hold = 1
# for i in range(30):
#     model = train_model("GOOG", str((date-timedelta(days=365)).date()), str(((date-timedelta(days=1)).date())), 64)
#     perf = evaluate(model, "GOOG", str(date.date()), str((date+timedelta(days=2)).date()))
#     money *= perf[0]
#     hold *= perf[1]
#     date += timedelta(days=1)
#
# print(money, hold)