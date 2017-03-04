from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

from .encoders import IdentityEncoder
from .utils import missing_mask
from ..utils import logger


class MLImputer(object):
    """generic class for imputing missing data using ML models.
    It uses a given regressor and a given classifier to impute numeric
    and categorical fields - respectively.
    For every column in the input dataframe a regressor (classifier) is
    trained on all the rows where this column is not missing. It is
    trained on all the remaining columns (if those other columns have
    missing values themselves, they are imputed using provided base_imputer.

    !!! it is assumed that missing values are encoded as NaN for float-y
    columns and as -1 for categorical (and boolean) columns. If they are not, a
    feature_encoder must be specified that will perform that encoding.
    """

    def __init__(self,
                 base_classifier=None,
                 base_regressor=None,
                 base_imputer=IdentityEncoder,
                 feature_encoder=IdentityEncoder(),
                 missing_features=None):
        """
        :param base_classifier: the sklearn-like classifier used to impute categorical columns
        :param base_regressor: the sklearn-like regressor used to impute continous columns
        :param base_imputer: the imputer used for the first crude run at imputation. It's only
            necessary when base_classifier or base_regressor can't handle missing values itself
            - which is the case for RandomForest. MasterExploder is a good default base_imputer
            in such cases. XGBoost regressors and classifiers don't need any base_imputer.
        :param feature_encoder: transformer (with .fit, .transform, .inverse_transform methods)
            that transforms a dataframe to a format where all categorical columns are integers
            (with missing values denoted by -1) and all continous columns are floats (with missing
            values denoted by np.NaN)
        :param missing_features: list of features the Imputer will train to impute. By default it
            trains to impute every column.
        """

        self.base_imputer = base_imputer
        self.feature_encoder = feature_encoder
        self.base_classifier = base_classifier
        self.base_regressor = base_regressor
        self.col2imputer = {}
        self.column_set = {}
        self.col2feats = {}
        self.col2type = {}
        self.missing_features = missing_features

    def __str__(self):
        return "MLImputer(%s, %s, %s, %s)" % (
            self.base_classifier(), self.base_regressor(), self.base_imputer(), self.feature_encoder)

    def fit(self, df, y=False):
        df = self.feature_encoder.fit(df).transform(df)
        self.column_set = set(df.columns)
        if self.missing_features is None:
            self.missing_features = self.column_set

        for col in self.missing_features:
            other_cols = sorted(self.column_set.difference({col}))
            self.col2feats[col] = other_cols

        for i, col in enumerate(self.missing_features):
            column = df.get(col)
            if np.issubdtype(column.dtype, np.floating):
                self.col2type[col] = 'numeric'
            elif all(x in {-1, 0, 1, False, True} for x in column):
                self.col2type[col] = 'boolean'
            else:
                self.col2type[col] = 'integer'

            logger.info("fitting MLImputer on %s column %s. %s column out of %s"
                        % (self.col2type[col], col, i + 1, len(self.column_set)))

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
                [('encoder', self.base_imputer()), ('imputer', model)])
            self.col2imputer[col] = imputer.fit(X, y)
            logger.info('column imputer fitted on %s column' % col)
        return self

    def transform(self, df, proba=False):
        df = self.feature_encoder.transform(df)
        result_dict = {}
        for col in self.column_set:
            column = df[col]
            column_copy = np.copy(column)
            if col in self.missing_features:
                feats = self.col2feats[col]
                missing = missing_mask(column)
                if missing.any():
                    if proba and self.col2type[col] == 'boolean':
                        predictions = (
                            self.col2imputer[col]
                            .predict_proba(df.get(feats)[missing].reset_index(drop=True))[:, 1]
                        )
                        column_copy += 0.0
                    else:
                        predictions = (
                            self.col2imputer[col]
                            .predict(df.get(feats)[missing].reset_index(drop=True))
                        )

                    column_copy[np.where(missing_mask(column))] = predictions

            result_dict[col] = column_copy

        transformed = self.feature_encoder.inverse_transform(
            pd.DataFrame(result_dict))
        transformed.index = df.index.copy()

        return transformed

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
