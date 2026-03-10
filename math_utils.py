import numpy as np

def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)

    # Numerically stable computation of the sigmoid function.
    # We want to make denominators close to 1 to avoid numerical instability.
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )