import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
from keras.models import Sequential, save_model, load_model
from keras.layers import LSTM, Dense, Dropout
import market_data

plt.style.use("bmh")


def main():
    df, close_scalar, scalar = market_data.get_ta_data("SPY", "2010-01-01", "2015-01-01")
    n_per_in = 90
    n_per_out = 10
    df = df[0:600]
    model = get_model(df.shape[1], n_per_in, n_per_out)
    # res = train_model(model, df)
    # save_model(model, "models/spy")
    model = load_model("models/spy")

    predicted = validator(model, df, close_scalar, n_per_in, n_per_out)
    actual = pd.DataFrame(close_scalar.inverse_transform(df[["Close"]]), index=df.index, columns=[df.columns[0]])

    plt.figure(figsize=(16, 6))

    plt.plot(predicted, label='Predicted')
    plt.plot(actual, label='Actual')

    plt.title(f"Predicted vs Actual Closing Prices")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

    forecast(model, df, close_scalar)


def train_model(model, df):
    n_per_in = model.input_shape[1]
    n_per_out = model.output_shape[1]
    x, y = split_sequence(df.to_numpy(), n_per_in, n_per_out)
    res = model.fit(x, y, epochs=20, batch_size=128, validation_split=0.1)
    return res



def get_model(n_features, n_per_in, n_per_out):
    model = Sequential()
    model.add(LSTM(90, activation="tanh", return_sequences=True, input_shape=(n_per_in, n_features)))
    model.add(Dropout(0.2))
    model.add(LSTM(30, activation="tanh", return_sequences=True))
    model.add(LSTM(60, activation="tanh"))
    model.add(Dense(n_per_out))
    model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
    return model


def forecast(model, df, close_scaler):
    n_per_in = model.input_shape[1]
    n_features = model.input_shape[2]
    # Predicting off of the most recent days from the original DF

    data = df.tail(n_per_in)
    data = np.array(data)
    yhat = model.predict(data.reshape((1, n_per_in, n_features)))

    # Transforming the predicted values back to their original format
    yhat = close_scaler.inverse_transform(yhat)[0]

    # Creating a DF of the predicted prices
    preds = pd.DataFrame(yhat,
                         index=pd.date_range(start=df.index[-1] + timedelta(days=1),
                                             periods=len(yhat),
                                             freq="B"),
                         columns=[df.columns[0]])

    # Number of periods back to plot the actual values
    pers = n_per_in

    # Transforming the actual values to their original price
    actual = pd.DataFrame(close_scaler.inverse_transform(df[["Close"]].tail(pers)),
                          index=df.Close.tail(pers).index,
                          columns=[df.columns[0]]).append(preds.head(1))

    # Printing the predicted prices
    print(preds)

    # Plotting
    plt.figure(figsize=(16, 6))
    plt.plot(actual, label="Actual Prices")
    plt.plot(preds, label="Predicted Prices")
    plt.ylabel("Price")
    plt.xlabel("Dates")
    plt.title(f"Forecasting the next {len(yhat)} days")
    plt.legend()
    plt.show()


def split_sequence(seq, n_steps_in, n_steps_out):
    x = []
    y = []

    for i in range(len(seq)):
        end = i + n_steps_in
        out_end = end + n_steps_out

        if out_end > len(seq):
            break

        seq_x = seq[i:end, :]
        seq_y = seq[end:out_end, 0]
        x.append(seq_x)
        y.append(seq_y)

    return np.array(x), np.array(y)


def visualize_training(results):
    history = results.history
    plt.figure(figsize=(16, 5))
    plt.plot(history['val_loss'])
    plt.plot(history['loss'])
    plt.legend(['val_loss', 'loss'])
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

    plt.figure(figsize=(16, 5))
    plt.plot(history['val_accuracy'])
    plt.plot(history['accuracy'])
    plt.legend(['val_accuracy', 'accuracy'])
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()


def validator(model, df, close_scaler, n_per_in, n_per_out):
    # do a rolling prediction

    predictions = pd.DataFrame(index=df.index, columns=[df.columns[0]])

    # for i in range(n_per_in, len(df) - n_per_in, n_per_out):
    #     # Creating rolling intervals to predict off of
    #     x = df[-i - n_per_in:-i]

    for i in range(0, len(df) - n_per_in, n_per_out):

        x = df[i:i+n_per_in]

        # Predicting using rolling intervals
        n_features = df.shape[1]
        yhat = model.predict(np.array(x).reshape((1, n_per_in, n_features)))

        # Transforming values back to their normal prices
        yhat = close_scaler.inverse_transform(yhat)[0]

        # DF to store the values and append later, frequency uses business days
        pred_df = pd.DataFrame(yhat,
                               index=pd.date_range(start=x.index[-1],
                                                   periods=len(yhat),
                                                   freq="B"),
                               columns=[x.columns[0]])

        # Updating the predictions DF
        predictions.update(pred_df)

    return predictions


def val_rmse(df1, df2):
    df = df1.copy()
    df["Close2"] = df2.Close
    df.dropna(inplace=True)
    df["diff"] = df.Close - df.Close2
    rms = (df[["diff"]]**2).mean()
    return float(np.sqrt(rms))


main()
