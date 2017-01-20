from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np
import pandas as pd
from ml_imputation import MLImputer


def DefaultImputer():
    return MLImputer(base_classifier=RandomForestClassifier, base_regressor=RandomForestRegressor)


def sample_dataset():
    N = -1
    NaN = np.NaN
    S = 1

    datax = pd.DataFrame(dict(
        a=[1, 1, 1, 1, 0, 0, 0, 1],
        b=[N, 0, 1, 0, N, 0, 1, 0],
        c=[1, 0, 0, N, N, 1, 0, 0],
        d=np.array([NaN, NaN, 1.0, NaN, NaN, 2.14, 0.0, NaN]),
        e=[3, N, N, 3, N, 3, 3, N]
    ))
    return datax
