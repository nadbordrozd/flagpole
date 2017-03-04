import pandas as pd

from dstk.imputation.utils import mask_missing
from dstk.imputation.bayes_imputation import BayesNetImputer
from dstk.pymc_utils import make_bernoulli, cartesian_child
import pymc


class RSWImputer(BayesNetImputer):
    """the famous Rain-Sprinkler-Wet bayesian network"""

    def construct_net(self, df):
        rain = make_bernoulli('rain', value=df.rain)
        sprinkler = make_bernoulli('sprinkler', value=df.sprinkler)
        sidewalk = cartesian_child('wet_sidewalk', parents=[rain, sprinkler],
                                   value=df.wet_sidewalk)

        model = pymc.Model([rain, sprinkler, sidewalk])
        return model


def get_rsw_full_data():
    # in this dataset 'rain' and 'sprinkler' are independent,
    # while 'wet_sidewalk' is true iff 'rain' OR 'sprinkler'
    return pd.DataFrame({
        'rain':         [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        'sprinkler':    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        'wet_sidewalk': [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1]
    })


def get_rsw_data_with_missing():
    return pd.DataFrame({
        'rain':         [-1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        'sprinkler':    [0, -1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        'wet_sidewalk': [0, 1, -1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1]
    })


def test_RSWImputer_imputes_stuff_with_MAP():
    imputer = RSWImputer(method='MAP')

    filled_data = imputer.fit_transform(get_rsw_data_with_missing())
    assert filled_data.equals(get_rsw_full_data())


def test_RSWImputer_imputes_stuff_with_MCMC():
    imputer = RSWImputer(method='MCMC')

    filled_data = imputer.fit_transform(get_rsw_data_with_missing())
    assert filled_data.equals(get_rsw_full_data())
