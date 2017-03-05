import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error
from time import time

from dstk.imputation.utils import mask_random_entries, missing_mask, is_floaty


def random_mask_data(data, missing_fraction, seed):
    np.random.seed(seed)
    missing_count = int(len(data) * missing_fraction)
    return pd.DataFrame({
        col: mask_random_entries(data[col], missing_count)
        for col in data.columns
    })


def benchmark(model, data, missing_fraction=0.1, seed=3):

    masked_data = random_mask_data(data, missing_fraction, seed)
    start = time()
    imputed = model.fit_transform(masked_data)
    end = time()
    results = {
        'ELAPSED_TIME': end - start
    }
    for col in data.columns:
        column = data.get(col)
        masked_column = masked_data.get(col)
        imputed_column = imputed.get(col)
        if imputed_column is None:
            results[col] = np.NaN
            continue
        mask = missing_mask(masked_column)

        actual = column[mask]
        predicted = imputed_column[mask]

        results[col] = (
            accuracy_score(actual, predicted) if not is_floaty(column)
            else mean_squared_error(actual, predicted)
        )

    return results
