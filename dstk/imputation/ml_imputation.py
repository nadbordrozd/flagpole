from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

from encoders import MasterExploder
from utils import missing_mask


class MLImputer(object):
    """generic class for imputing missing data with using ML models.
    It uses a given regressor and a given classifier to impute numeric
    and categorical fields - respectively.
    For every column in the input dataframe a regressor (classifier) is
    trained on all the rows where this column is not missing. It is
    trained on all the remaining columns (if those other columns have
    missing values themselves, they are imputed simply using median
    (for numeric fields) or encoded as another class (for categorical).

    !!! it is assumed that missing values are encoded as NaN for float-y
    columns and as -1 for categorical (and boolean) columns.
    For example usage see test_MLImputer
    """

    def __init__(
            self,
            base_classifier=None,
            base_regressor=None,
            base_encoder=MasterExploder):
        self.base_classifier = base_classifier
        self.base_encoder = base_encoder
        self.base_regressor = base_regressor
        self.columns = None
        self.col2imputer = {}
        self.col2feats = {}
        self.col2type = {}

    def __str__(self):
        return "MLImputer(%s, %s)" % (self.base_classifier(), self.base_regressor)

    def fit(self, df, y=False):

        self.column_set = set(df.columns)

        for col in self.column_set:
            other_cols = sorted(self.column_set.difference({col}))
            self.col2feats[col] = other_cols

        for col in self.column_set:
            column = df.get(col)
            if np.issubdtype(column.dtype, np.floating):
                self.col2type[col] = 'numeric'
            elif all(x in {-1, 0, 1, False, True} for x in column):
                self.col2type[col] = 'boolean'
            else:
                self.col2type[col] = 'integer'

            feats = self.col2feats[col]
            if np.issubdtype(column.dtype, np.floating):
                model = self.base_regressor()
            else:
                model = self.base_classifier()

            X = df.get(feats)[~missing_mask(column)].reset_index(drop=True)
            y = column[~missing_mask(column)].values
            if len(y) == 0:
                raise ValueError(
                    'need at least 1 nonmissing value to train imputer')

            imputer = Pipeline(
                [('encoder', self.base_encoder()), ('imputer', model)])
            self.col2imputer[col] = imputer.fit(X, y)
        return self

    def transform(self, df, proba=False):
        result_dict = {}
        for col, feats in self.col2feats.items():
            column = df[col]
            column_copy = np.copy(column)

            missing = missing_mask(column)
            if missing.any():
                if proba and self.col2type[col] == 'boolean':
                    predictions = self.col2imputer[col] \
                                      .predict_proba(
                        df.get(feats)[missing]
                            .reset_index(drop=True))[:, 1]
                    column_copy = column_copy + 0.0
                else:
                    predictions = self.col2imputer[col] \
                        .predict(
                        df.get(feats)[missing]
                            .reset_index(drop=True))

                column_copy[np.where(missing_mask(column))] = predictions

            result_dict[col] = column_copy

        return pd.DataFrame(result_dict)


def test_MLImputer():
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

    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    result = (
        MLImputer(RandomForestClassifier, RandomForestRegressor)
            .fit(datax)
            .transform(datax, proba=True)
    )

    assert set(result.columns) == {'a', 'b', 'c', 'd', 'e'}
    assert result.get('a').dtype == np.int64
    assert result.get('b').dtype == np.float64
    assert result.get('c').dtype == np.float64
    assert result.get('d').dtype == np.float64
    assert result.get('e').dtype == np.int64

    result = (
        MLImputer(RandomForestClassifier, RandomForestRegressor)
            .fit(datax)
            .transform(datax)
    )

    assert set(result.columns) == {'a', 'b', 'c', 'd', 'e'}
    assert result.get('a').dtype == np.int64
    assert result.get('b').dtype == np.int64
    assert result.get('c').dtype == np.int64
    assert result.get('d').dtype == np.float64
    assert result.get('e').dtype == np.int64
