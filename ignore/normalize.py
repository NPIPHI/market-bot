import numpy as np


def chunck(data, period):
    ret = np.ndarray((data.shape[0] - period, period, data.shape[1], ))
    for i in range(data.shape[0] - period):
        ret[i] = data[i:i + period]

    return ret


def normalize(data):
    ret = data.copy()
    for i in range(len(ret)):
        max = ret[i].max(axis=0)
        min = ret[i].min(axis=0)
        delta = max - min
        for j in range(len(delta)):
            if delta[j] == 0:
                delta[j] = 1

        ret[i] = (ret[i] - min) / delta
    return ret
