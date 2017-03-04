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
import pymc

from dstk.imputation.utils import is_floaty, missing_indices


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

    for col in data:
        if col not in sampler.observed_stochastics:
            predicted[col] = data[col]

    return pd.DataFrame(predicted)


class BayesNetImputer(object):

    def __init__(self, method='MAP', iter=500, burn=300, thin=2):
        """superclass of bayes-net-based models for imputing missing data

        To use this imputer, it should be subclassed and the methods 'construct_net' should be
        implemented - see examples in unit tests.

        The interface is the same as for other imputers in this package:

        data_with_missing_values_filled_in = imputer.fit(data).transform(data)

        This model fits and transforms data all in one go - inside the 'transform' function,
        therefore 'fit' is unnecessary and only there for consistency with other models.

        Two imputation methods are supported:
        - 'MCMC' - samples a fixed number of times from the posterior and returns mean (for numeric)
            or mode (for categorical) for each missing value
        - 'MAP' - directly finds the value of each missing variable tha maximizes posterior
            probability

        In theory these methods should be equivalent for categorical variables, but MAP should be
        much faster.

        :param method: either 'MAP' or 'MCMC', defaults to 'MAP'. If 'MAP' then, there will be no
            sampling, just optimisation and 'iter', 'burn', 'thin' parameters will have no effect.
        :param iter: total number iteratoins (only for 'MCMC' method)
        :param burn: number of initial iterations to discard (only for 'MCMC')
        :param thin: discard all except every nth iteration where n=thin
        """
        if method not in ['MAP', 'MCMC']:
            raise ValueError("only 'MAP' and 'MCMC' methods are supported")
        self.method = method
        self.iter = iter
        self.burn = burn
        self.thin = thin

    def fit(self, df, y=None):
        return self

    def construct_net(self, df):
        """fit your bayes net here and return pymc.Model"""
        raise NotImplementedError()

    def sample(self, model):
        sampler = pymc.MCMC(model)
        sampler.sample(iter=self.iter, burn=self.burn, thin=self.thin)
        return sampler

    def transform(self, df):
        model = self.construct_net(df)
        if self.method == 'MAP':
            pymc.MAP(model).fit()
            transformed_df = df.copy()
            for node in model.stochastics:
                name = str(node)
                transformed_df[name] = node.value + 0
            return transformed_df
        else:
            sampler = self.sample(model)
            return predict(sampler, df)

    def fit_transform(self, X, y=None):
        return self.transform(X)
