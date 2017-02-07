"""
This is a template for missing value imputation with bayesian networks. By their nature, bayesian
nets require to be constructed separately for every new usecase - so this module can't provide a net
that would be usable right away for any future application. What it does provide is:
 - a template classBayesImputer that conforms to the same interface as other imputers in this
   package
 - a few utility functions specific to missing data (in addition to the stuff in dstk.pymc_utils)
 - example usage - see tests at the bottom
"""

import pandas as pd
import numpy as np

from dstk.imputation.utils import is_floaty, mask_missing, missing_indices


def predict(sampler, data):
    predicted = {}
    for node in sampler.observed_stochastics:
        feature_name = str(node)
        values = data[feature_name]
        missing_inds = missing_indices(values)
        if len(missing_inds):
            trace = sampler.trace(feature_name)[:]

            mean_posterior = values[:]
            for samples, ind in zip(trace.T, missing_inds):
                if is_floaty(samples):
                    mean_posterior[ind] = samples.mean()
                else:
                    mean_posterior[ind] = np.bincount(samples).argmax()

            predicted[feature_name] = mean_posterior
        else:
            predicted[feature_name] = values

    return pd.DataFrame(predicted)


class BayesNetImputer(object):
    """
    superclass of bayes-net based models that for transforming data

    the interface is the same as all the other imputers:
    data_with_missing_filled_in = imputer.fit(data).predict(data)

    As model fitting and prediction are both done in one go - sampling, it is not possible to fit
    the model on one dataset and predict for another. This is why only the 'transform' method
    of this class does any useful work while 'fit' is empty. 'fit' is kept here for consistency with
    other types of imputers.
    """

    def __init__(self, iter=500, burn=300, thin=2):
        self.iter = iter
        self.burn = burn
        self.thin = thin
        self.sampler = None

    def fit(self, df, y=None):
        return self

    def construct_net(self, df):
        """fit your bayes net here and return sampler"""
        raise NotImplementedError()

    def sample(self):
        self.sampler.sample(iter=self.iter, burn=self.burn, thin=self.thin)

    def transform(self, df):
        self.sampler = self.construct_net(df)
        self.sample()
        return predict(self.sampler, df)


def test_RSWImputer_imputes_stuff():
    from dstk.pymc_utils import make_bernoulli, cartesian_child
    import pymc

    # in this dataset 'rain' and 'sprinkler' are independent,
    # while 'wet_sidewalk' is true iff either 'rain' or 'sprinkler' or both
    full_data = pd.DataFrame({
        'rain': [0, 0, 1, 1, 1, 1, 0, 0],
        'sprinkler': [0, 1, 1, 0, 1, 0, 1, 0],
        'wet_sidewalk': [0, 1, 1, 1, 1, 1, 1, 0],
    })

    data = pd.DataFrame({
        'rain': [0, 0, 1, 1, 1, -1, 0, -1],
        'sprinkler': [0, 1, 1, 0, 1, 0, -1, -1],
        'wet_sidewalk': [0, 1, 1, 1, 1, 1, 1, 0],
    })

    class RSWImputer(BayesNetImputer):

        def construct_net(self, df):
            rain_data = mask_missing(df.rain)
            sprinkler_data = mask_missing(df.sprinkler)
            sidewalk_data = mask_missing(df.wet_sidewalk)

            rain = make_bernoulli('rain', value=rain_data)
            sprinkler = make_bernoulli('sprinkler', value=sprinkler_data)
            sidewalk = cartesian_child('wet_sidewalk', parents=[rain, sprinkler],
                                       value=sidewalk_data)

            model = pymc.Model([rain, sprinkler, sidewalk])
            sampler = pymc.MCMC(model)
            return sampler

    imputer = RSWImputer()

    filled_data = imputer.fit(data).transform(data)
    assert filled_data.equals(full_data)
