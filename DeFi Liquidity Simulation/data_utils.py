# data_utils.py

import numpy as np

def generate_synthetic_price_series(num_steps=1000,
                                    start_price=1.0,
                                    drift=0.0,
                                    volatility=0.01,
                                    random_seed=None):
    """
    Generates a synthetic price series using geometric Brownian motion.
    If 'random_seed' is provided, we set the global NumPy RNG seed temporarily,
    which makes the output reproducible. If 'random_seed' is None,
    it uses whatever the current global random state is (fully random).
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    prices = np.zeros(num_steps)
    prices[0] = start_price
    for t in range(1, num_steps):
        shock = np.random.normal(0, 1)
        prices[t] = prices[t - 1] * np.exp((drift - 0.5 * volatility ** 2) + volatility * shock)

    return prices
