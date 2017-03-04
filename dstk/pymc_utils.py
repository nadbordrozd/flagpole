import pymc
import numpy as np
from itertools import product

COEFFS_PREFIX = 'CF: '


def set_levels_count(cat_var, levels):
    """sets the number of levels for a categorical pymc variable"""
    cat_var.number_of_levels = levels
    return cat_var


def get_levels_count(cat_var):
    """takes categorical variable and returns the number of categories"""
    return cat_var.number_of_levels


def make_categorical(name, levels, value=None, N=None,
                     return_coeffs=False, fixed={}):
    """ creates a Bernoulli random variable with a Dirichlet parent

    :param name: name of the variable
    :param levels: integer - how many levels does the variable have
    :param value: optional - list of observed values of the variable. Must consist of integers
        from 0 to levels - 1. May be a masked array - if the variable has missing values
    :param N: size of the variable (number of values). Either N or value must be specified
    :param return_coeffs: if true, will return the parent Beta variable as well as the bernoulli
        child. False by defaut.
    :param fixed: optional dictionary of values of coefficients to be fixed.
    :return: Categorical pymc random variable, or (if return_coeffs == True) a tuple
        (categorical variable; a list with a single element - the Dirichlet parent)
    """
    if value is None and N is None:
        raise ValueError('either "value" or "N" must be specified')
    N = N or len(value)
    coeff_name = COEFFS_PREFIX + 'p(%s)' % name
    if coeff_name in fixed:
        probs = fixed[coeff_name]
        parent = list(probs) + [1 - sum(probs)]
    else:
        parent = pymc.Dirichlet(coeff_name, theta=[1] * levels)

    if value is None:
        child = pymc.Categorical(name, p=parent, value=np.zeros(N))
    else:
        child = pymc.Categorical(name, p=parent, observed=True, value=value)

    set_levels_count(child, levels)

    if return_coeffs:
        return child, [parent]
    else:
        return child


def make_bernoulli(name, value=None, N=None, return_coeffs=False, fixed={}):
    """ creates a Bernoulli random variable with a uniform parent

    :param name: name of the variable
    :param value: optional - list of observed values of the variable. May be a masked array - if
        the variable has missing values
    :param N: size of the variable (number of values). Either N or value must be specified
    :param return_coeffs: if true, will return the parent Beta variable as well as the bernoulli
        child. False by defaut.
    :param fixed: optional dictionary of values of coefficients to be fixed.
    :return: Bernoulli pymc random variable, or (if return_coeffs == True) a tuple
        (bernoulli variable, a list with a single element - the Beta parent of the bernoulli)
    """
    if value is None and N is None:
        raise ValueError('either "value" or "N" must be specified')

    parent_name = COEFFS_PREFIX + 'p(%s)' % name
    parent = fixed.get(parent_name, pymc.Beta(parent_name, 1, 1))

    if value is None:
        child = pymc.Bernoulli(name, p=parent, value=np.zeros(N))
    else:
        child = pymc.Bernoulli(name, p=parent, observed=True, value=value)

    set_levels_count(child, 2)
    if return_coeffs:
        return child, [parent]
    else:
        return child


def cartesian_bernoulli_child(
        name, parents, value=None, N=None, return_coeffs=False, fixed={}):
    if value is None and N is None:
        raise ValueError('either "value" or "N" must be specified')

    ranges = [range(get_levels_count(p)) for p in parents]
    parents2index = {}
    coeffs = []
    for i, parent_vals in enumerate(product(*ranges)):
        parents2index[parent_vals] = i
        parents_repr = ' '.join('%s=%s' % (parent, v) for parent, v in zip(parents, parent_vals))
        coeff_name = COEFFS_PREFIX + 'p(%s | %s)' % (name, parents_repr)
        coeff = fixed.get(coeff_name, pymc.Uniform(coeff_name, 0, 1))
        coeffs.append(coeff)

    intify = lambda x: tuple(map(int, x))

    @pymc.deterministic
    def child_prob(parents=parents, coeffs=coeffs):
        return np.array([
            coeffs[parents2index[intify(parent_vals)]]
            for parent_vals in zip(*parents)])

    child_prob.__name__ = 'p(%s)' % name

    if value is None:
        child = pymc.Bernoulli(name, p=child_prob, value=np.zeros(N))
    else:
        child = pymc.Bernoulli(name, p=child_prob, value=value, observed=True)
    set_levels_count(child, 2)

    if return_coeffs:
        return child, coeffs + [child_prob]
    else:
        return child


def cartesian_categorical_child(
        name, parents, levels, value=None, N=None, return_coeffs=False, fixed={}):
    if value is None and N is None:
        raise ValueError('either "value" or "N" must be specified')

    ranges = [range(get_levels_count(p)) for p in parents]
    parents2index = {}
    coeffs = []
    for i, parent_vals in enumerate(product(*ranges)):
        parents2index[parent_vals] = i
        parents_repr = ' '.join('%s=%s' % (parent, v) for parent, v in zip(parents, parent_vals))
        coeff_name = COEFFS_PREFIX + 'p(%s | %s)' % (name, parents_repr)
        coeff = fixed.get(coeff_name, pymc.Dirichlet(
            coeff_name, theta=[1] * levels))
        coeffs.append(coeff)

    intify = lambda x: tuple(map(int, x))

    @pymc.deterministic
    def child_prob(parents=parents, coeffs=coeffs):
        probs = np.array([
            coeffs[parents2index[intify(parent_vals)]]
            for parent_vals in zip(*parents)])
        remainders = 1 - probs.sum(axis=1)
        remainders = remainders.reshape((len(remainders), 1))
        return np.hstack([probs, remainders])

    child_prob.__name__ = 'p(%s)' % name

    if value is None:
        child = pymc.Categorical(name, p=child_prob, value=np.zeros(N))
    else:
        child = pymc.Categorical(
            name, p=child_prob, value=value, observed=True)
    set_levels_count(child, levels)

    if return_coeffs:
        return child, coeffs + [child_prob]
    else:
        return child


def cartesian_child(name, parents, levels=2, value=None,
                    N=None, return_coeffs=False, fixed={}):
    if levels > 2:
        return cartesian_categorical_child(
            name, parents, levels, value, N, return_coeffs, fixed)
    else:
        return cartesian_bernoulli_child(
            name, parents, value, N, return_coeffs, fixed)


def _linearised(parent, child_name, return_coeffs=False, fixed={}):
    if hasattr(parent, 'number_of_levels'):
        levels = get_levels_count(parent)
        coeffs = []
        for i in range(1, levels):
            coeff_name = COEFFS_PREFIX + \
                '(%s==%s)->%s' % (str(parent), i, child_name)
            coeff = fixed.get(coeff_name, pymc.Normal(coeff_name, 0, 0.5))
            coeffs.append(coeff)

        @pymc.deterministic
        def linearised(par=parent, cf=coeffs):
            return [cf[val - 1] for val in par]
    else:
        coeffs = [pymc.Normal('%s->%s' % (str(parent), child_name), 0, 0.5)]
        linearised = parent * coeffs[0]
    linearised.__name__ = 'lin(%s for %s)' % (str(parent), child_name)

    if return_coeffs:
        return linearised, coeffs
    else:
        return linearised


def _linearised_many(parents, child_name, return_coeffs=False, fixed={}):
    theta_name = COEFFS_PREFIX + 'bias(%s)' % child_name
    theta = fixed.get(theta_name, pymc.Normal(theta_name, 0, 0.5))
    all_coeffs = [theta]
    for parent in parents:
        lin, coeffs = _linearised(parent, child_name, True, fixed)
        theta = theta + lin
        all_coeffs.extend(coeffs)

    if return_coeffs:
        return theta, all_coeffs
    else:
        return theta


def logistic_bernoulli_child(name, parents, value=None,
                             N=None, return_coeffs=False, fixed={}):
    N = N or len(value)
    theta, all_coeffs = _linearised_many(parents, name, True, fixed)

    child_prob = pymc.InvLogit('p(%s)' % name, theta)
    if value is None:
        child = pymc.Bernoulli(name, p=child_prob, value=np.zeros(N))
    else:
        child = pymc.Bernoulli(name, p=child_prob, value=value, observed=True)
    set_levels_count(child, 2)

    if return_coeffs:
        return child, all_coeffs
    else:
        return child


def logistic_categorical_child(name, parents, levels, value=None,
                               N=None, return_coeffs=False, fixed={}):
    N = N or len(value)
    all_coeffs, one_v_all = [], []
    for i in range(levels):
        theta, coeffs = _linearised_many(
            parents, '%s==%s' % (name, i), True, fixed)
        level_prob = pymc.InvLogit('p(%s==%s)' % (name, i), theta)
        one_v_all.append(level_prob)
        all_coeffs.extend(coeffs)

    @pymc.deterministic
    def child_prob(level_probs=one_v_all):
        ret = [np.array(probs) / sum(probs)
               for probs in zip(*level_probs)]
        return ret

    child_prob.__name__ = 'p(%s)' % name

    if value is None:
        child = pymc.Categorical(name, p=child_prob, value=np.zeros(N))
    else:
        child = pymc.Categorical(
            name, p=child_prob, value=value, observed=True)
    set_levels_count(child, levels)

    if return_coeffs:
        return child, all_coeffs
    else:
        return child


def logistic_child(name, parents, levels=2, value=None, N=None,
                   return_coeffs=False, fixed={}):
    if levels > 2:
        return logistic_categorical_child(name, parents, levels,
                                          value, N, return_coeffs, fixed)
    else:
        return logistic_bernoulli_child(
            name, parents, value, N, return_coeffs, fixed)


def sample_coeffs(pymc_model):
    """samples model coeffs from their priors, returns as a dictionary"""
    sampler = pymc.MCMC(pymc_model)
    # no need for burn-in
    sampler.sample(iter=1)

    result = {}
    for x in sampler.stochastics:
        name = str(x)
        if name.startswith(COEFFS_PREFIX):
            val = sampler.trace(name)[-1:][0]
            result[name] = val
    return result
