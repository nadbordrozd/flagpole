import numpy as np
import pandas as pd
from xgboost.sklearn import XGBClassifier, XGBRegressor
from pandas.util.testing import assert_frame_equal
from .ml_imputation import MLImputer
from .encoders import StringFeatureEncoder


def test_MLImputer_imputes_missing_values():
    N = -1
    NaN = np.NaN

    datax = pd.DataFrame(dict(
        a=['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B'],
        b=[N, 0, 1, 0, N, 0, 1, 0],
        c=[1, 0, 0, N, N, 1, 0, 0],
        d=np.array([NaN, NaN, 1.0, NaN, NaN, 2.14, 0.0, NaN]),
    ))

    result = (
        MLImputer(XGBClassifier, XGBRegressor, feature_encoder=StringFeatureEncoder())
            .fit(datax)
            .transform(datax)
    )

    expected = pd.DataFrame({
        'a': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B'],
        'b': [0, 0, 1, 0, 0, 0, 1, 0],
        'c': [1, 0, 0, 0, 0, 1, 0, 0],
        'd': [0.002960, 2.130146, 1.000000, 0.002960, 2.130146, 2.140000, 0.0, 2.130146]
    })

    assert_frame_equal(result, expected, check_less_precise=True)
