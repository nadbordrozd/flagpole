from dstk.pymc_utils import make_bernoulli, make_categorical,\
    cartesian_bernoulli_child, cartesian_categorical_child, COEFFS_PREFIX, sample_coeffs
import pymc
import numpy as np


def pad(s, width):
    return s + ' ' * (width - len(s))


def test_make_bernoulli_returns_variable_with_beta_parent():
    name = 'rosebud'
    bernoulli_var = make_bernoulli(name, value=[0, 1, 1, 0])
    parent = bernoulli_var.parents['p']

    assert bernoulli_var.observed
    assert not parent.observed
    assert str(parent) == COEFFS_PREFIX + 'p(%s)' % name
    assert isinstance(parent, pymc.distributions.Beta)


def test_make_categorical_returns_variable_with_dirichlet_parent():
    name = 'genialon'
    cat_var = make_categorical(name, 4, value=[1, 2, 3, 0, 1])
    parent = cat_var.parents['p']
    assert cat_var.observed
    assert not parent.observed
    assert str(parent) == COEFFS_PREFIX + 'p(%s)' % name
    assert isinstance(parent, pymc.distributions.Dirichlet)


def test_cartesian_bernoulli_child_creates_correctly_named_coefficients():
    mum = make_bernoulli('mum', N=1)
    dad = make_bernoulli('dad', N=1)
    child, coeffs = cartesian_bernoulli_child('child', [mum, dad], N=1, return_coeffs=True)
    coeff_names = {str(coeff) for coeff in coeffs}
    assert coeff_names == {
        'CF: p(child | mum=0 dad=0)',
        'CF: p(child | mum=1 dad=1)',
        'CF: p(child | mum=1 dad=0)',
        'CF: p(child | mum=0 dad=1)',
        'p(child)'
    }

    single_parent = make_bernoulli('single_parent', N=1)
    child, coeffs = cartesian_bernoulli_child('child', [single_parent], N=1, return_coeffs=True)
    coeff_names = {str(coeff) for coeff in coeffs}
    assert coeff_names == {
        'p(child)',
        'CF: p(child | single_parent=0)',
        'CF: p(child | single_parent=1)'
    }


def test_cartesian_categorical_child_creates_correctly_named_coefficients():
    mum = make_bernoulli('mum', N=1)
    dad = make_bernoulli('dad', N=1)
    child, coeffs = cartesian_categorical_child(
        'child', [mum, dad], levels=4, N=1, return_coeffs=True)

    coeff_names = {str(coeff) for coeff in coeffs}
    assert coeff_names == {
        'CF: p(child | mum=0 dad=0)',
        'CF: p(child | mum=1 dad=1)',
        'CF: p(child | mum=1 dad=0)',
        'CF: p(child | mum=0 dad=1)',
        'p(child)'
    }


def test_cartesian_bernoulli_child():
    # define the model with no data just to sample all the coefficients from
    # their priors
    has_garden = make_bernoulli('has_garden', N=1)
    is_big = make_bernoulli('is_big', N=1)
    is_green = cartesian_bernoulli_child('is_green', [is_big, has_garden], N=1)
    model = pymc.Model([has_garden, is_big, is_green])

    coeff_values = sample_coeffs(model)

    # define identical model again but fix coefficient values
    has_garden = make_bernoulli('has_garden', N=1, fixed=coeff_values)
    is_big = make_bernoulli('is_big', N=1, fixed=coeff_values)
    is_green = cartesian_bernoulli_child(
        'is_green', [is_big, has_garden], N=1, fixed=coeff_values)
    fx_model = pymc.Model([has_garden, is_big, is_green])
    # sample from the model with fixed coefficients
    fx_sampler = pymc.MCMC(fx_model)
    fx_sampler.sample(iter=2000)

    has_garden_sample = fx_sampler.trace('has_garden')[:]
    is_big_sample = fx_sampler.trace('is_big')[:]
    is_green_sample = fx_sampler.trace('is_green')[:]

    # define identical model again but fix coefficient values
    has_garden, cfs1 = make_bernoulli(
        'has_garden', value=has_garden_sample, return_coeffs=True)
    is_big, cfs2 = make_bernoulli(
        'is_big', value=is_big_sample, return_coeffs=True)
    is_green, cfs3 = cartesian_bernoulli_child(
        'is_green', [is_big, has_garden], value=is_green_sample, return_coeffs=True)

    model = pymc.Model(cfs1 + cfs2 + cfs3)
    sampler = pymc.MCMC(model)
    sampler.sample(iter=2000, burn=1000)

    for pymc_var in sampler.stochastics:
        name = str(pymc_var)
        mean_posterior = sampler.trace(name)[:].mean()
        actual = coeff_values[name]
        print "%s  %.3f   %.3f" % (pad(name, 30), mean_posterior, actual)
        assert np.isclose(mean_posterior, actual, rtol=0.1, atol=0.1)


def test_cartesian_bernoulli_child_of_categorical_parent():
    coeffs = {
        'CF: p(feeling_sick)': 0.55,
        'CF: p(day_of_week)': [0.013, 0.626, 0.039, 0.108, 0.134, 0.019],
        'CF: p(staying_home | day_of_week=5 feeling_sick=1)': 0.240,
        'CF: p(staying_home | day_of_week=3 feeling_sick=1)': 0.467,
        'CF: p(staying_home | day_of_week=4 feeling_sick=1)': 0.603,
        'CF: p(staying_home | day_of_week=0 feeling_sick=0)': 0.974,
        'CF: p(staying_home | day_of_week=6 feeling_sick=1)': 0.331,
        'CF: p(staying_home | day_of_week=6 feeling_sick=0)': 0.009,
        'CF: p(staying_home | day_of_week=2 feeling_sick=1)': 0.317,
        'CF: p(staying_home | day_of_week=0 feeling_sick=1)': 0.900,
        'CF: p(staying_home | day_of_week=2 feeling_sick=0)': 0.651,
        'CF: p(staying_home | day_of_week=1 feeling_sick=0)': 0.603,
        'CF: p(staying_home | day_of_week=1 feeling_sick=1)': 0.954,
        'CF: p(staying_home | day_of_week=3 feeling_sick=0)': 0.856,
        'CF: p(staying_home | day_of_week=5 feeling_sick=0)': 0.606,
        'CF: p(staying_home | day_of_week=4 feeling_sick=0)': 0.828
    }
    np.random.seed = 1
    # define the model with fixed *coefficients*
    day_of_week = make_categorical('day_of_week', levels=7, N=1, fixed=coeffs)
    feeling_sick = make_bernoulli('feeling_sick', N=1, fixed=coeffs)
    staying_home = cartesian_bernoulli_child('staying_home', [day_of_week, feeling_sick], N=1,
                                             fixed=coeffs)
    model = pymc.Model([day_of_week, feeling_sick, staying_home])

    # sample data from the model
    sampler = pymc.MCMC(model)
    sampler.sample(iter=500)
    day_of_week_val = sampler.trace('day_of_week')[:]
    feeling_sick_val = sampler.trace('feeling_sick')[:]
    staying_home_val = sampler.trace('staying_home')[:]

    # define the model again this time with fixed *data*
    day_of_week, c1 = make_categorical('day_of_week', levels=7, value=day_of_week_val,
                                       return_coeffs=True)
    feeling_sick, c2 = make_bernoulli(
        'feeling_sick', value=feeling_sick_val, return_coeffs=True)
    staying_home, c3 = cartesian_bernoulli_child('staying_home',
                                                 [day_of_week, feeling_sick],
                                                 value=staying_home_val, return_coeffs=True)
    model = pymc.Model(
        [day_of_week, feeling_sick, staying_home] + c1 + c2 + c3)
    sampler = pymc.MCMC(model)
    sampler.sample(iter=500, burn=300)

    cname = 'CF: p(staying_home | day_of_week=1 feeling_sick=0)'
    print cname
    assert np.isclose(coeffs[cname], sampler.trace(cname)[:].mean(), atol=0.3)
    for c in coeffs:
        print c, coeffs[c], sampler.trace(c)[:].mean()
