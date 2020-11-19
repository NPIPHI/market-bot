import market_data
from keras import models
from keras import layers
import numpy as np
from ignore.normalize import chunck, normalize


def train_model(ticker, start, end, period):
    data = market_data.get_data(ticker, start, end)

    train_count = int((len(data) - period) * 0.9)
    test_count = len(data) - period - train_count

    gains = np.ndarray((train_count+test_count,))
    train_pasts = np.ndarray((train_count, period, data.shape[1]))
    train_gains = np.ndarray((train_count,))
    test_pasts = np.ndarray((test_count, period, data.shape[1]))
    test_gains = np.ndarray((test_count,))

    pasts = normalize(chunck(data, period))
    for i in range(train_count+test_count):
        gains[i] = data.Close[i+period] > data.Close[i+period-1]

    mapping = np.asarray(range(test_count+train_count))
    np.random.shuffle(mapping)

    for mapping, i in zip(mapping, range(len(mapping))):
        if(i < train_count):
            train_pasts[i] = pasts[mapping]
            train_gains[i] = gains[mapping]
        else:
            test_pasts[i-train_count] = pasts[mapping]
            test_gains[i-train_count] = gains[mapping]

    network = models.Sequential()
    network.add(layers.Conv1D(32, (16,), activation="relu", input_shape=(period, 7)))
    network.add(layers.MaxPool1D())
    network.add(layers.Flatten())
    # network.add(layers.Flatten(input_shape=(period, 7)))
    network.add(layers.Dense(64, activation="relu"))
    network.add(layers.Dense(64, activation="relu"))
    network.add(layers.Dense(1, activation="sigmoid"))
    network.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

    network.fit(train_pasts, train_gains, epochs=30)

    print(network.evaluate(test_pasts, test_gains))

    return network
