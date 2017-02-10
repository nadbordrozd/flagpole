from xgboost.sklearn import XGBClassifier, XGBRegressor
import numpy as np
import pandas as pd
from ml_imputation import MLImputer
from encoders import MasterExploder, StringFeatureEncoder
from bayes_imputation import BayesNetImputer
from utils import mask_missing


def DefaultImputer(missing_string_marker='UNKNOWN'):
    return MLImputer(
        base_classifier=XGBClassifier,
        base_regressor=XGBRegressor,
        feature_encoder=StringFeatureEncoder(ml_imputation=missing_string_marker))


def sample_dataset():
    N = -1
    NaN = np.NaN

    datax = pd.DataFrame(dict(
        a=[1, 1, 1, 1, 0, 0, 0, 1],
        b=[N, 0, 1, 0, N, 0, 1, 0],
        c=[1, 0, 0, N, N, 1, 0, 0],
        d=np.array([NaN, NaN, 1.0, NaN, NaN, 2.14, 0.0, NaN]),
        e=[3, N, N, 3, N, 3, 3, N]
    ))
    return datax


def wet_dataset():
    data = pd.DataFrame({
        'rain': [0, 0, 1, 1, 1, -1, 0, -1],
        'sprinkler': [0, 1, 1, 0, 1, 0, 1, -1],
        'wet_sidewalk': [0, 1, 1, 1, 1, 1, -1, 0],
        'some_numeric': [1.1, np.NaN, 0.2, -0.4, 0.1, 0.2, 0.0, 3.9],
        'some_string': ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'UNKNOWN']
    })
    return data
