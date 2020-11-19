from keras import models
from evaluate import evaluate

model = models.load_model("models/wmtconv")

print(evaluate(model, "WMT", "2010-01-01", "2010-02-01"))