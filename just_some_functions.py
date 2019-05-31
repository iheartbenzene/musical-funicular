import numpy as np
import scipy
import statsmodels.api as sm
import pandas as pd

from math import gamma


def factorial(n):
    if n < 0 and n%2 == 0:
        print("Please enter a non-negative integer.")
    elif n < 0 and n%2 != 0:
        print("Please enter a non-negative integer.")
    elif n == 0 or n == 1:
        return 1
    return n * factorial(n-1)


class NegativeBinomial(Poisson, Gamma):
    '''
    A negative binomial family when the Poisson distribution 
    has a Gamma distributed mean.

    '''
    def negative_binomial(self, k, r, nu):
        '''
        The number of sucesses.
        The number of failures.
        The mean of the distribution, mu.
        '''

        mu = np.exp(nu)
        p = r / (mu + r)
        q = mu / (mu + r)
        return (gamma(k+r)/(factorial(k)*gamma(r))) * (p**r) * (q**k)


def tweedie(n, p, phi, mu):
    '''
    n is the number of points to be generated
    p is a constant between 1 and 2
    phi is the dispersion
    mu is the mean
    '''

    np.random.seed(seed=32, version=2)
    if(p==2):
        print('1 < p < 2')
        pass
    rt = np.full(n, np.nan)
    lambdaa = mu ** (2-p)


# Lemma: calculate probabilities
from math import lgamma
from numba import jit

@jit
def h(a, b, c, d):
    number = lgamma(a + c) + lgamma(a + b) + lgamma(c + d) + lgamma(b + d)
    density = lgamma(a) + lgamma(b) + lgamma(c) + lgamma(d) + lgamma(a + b + c + d)
    return np.exp(number - density)

@jit
def g0(a, b, c):
    return (lgamma(a + c) + lgamma(a + c) - lgamma(a + b + c) + lgamma(a)

@jit
def h_iter(a, b, c, d):
    while d > 1:
        d -= 1
        yield h(a, b, c, d) / d

def g(a, b, c, d):
    return g0(a, b, c) + sum(h_iter(a, b, c, d))

def calculate_probabilities(beta1, beta2):
    return g(beta1.args[0], beta1.args[1], beta2.args[0], beta2.args[1])

#A/B Test
from scipy.stats import beta

import pandas as pd
import seaborn as sns
import numpy as np

'''
relate whatever text to a numerical value in control and test sets 
worst case scenario can be done via dataframe/index
'''

impressions_control, conversions_control = impressions control number, converstions control number
impressions_test, conversions_test = impressions test number, conversions test number

a_control, b_control = converstions_control, impressions_control - conversions_control + 1
beta_control = beta(a_control, b_control)
a_test, b_test = conversions_test, impressions_test - conversions_test + 1
beta_test = beta(a_test, b_test)

lift = (beta_test.mean() - beta_control.mean()) / beta_control.mean()
probability = calculate_probablities(beta_test, beta_control)


# for visuals
values_control = np.random.beta(a_control, b_control, 1e6)
values_test = np.random.beta(a_control, b_control, 1e6)
values = np.vstack([values_control, values_test]).T

limit = 4e-4

data_frame = pd.Dataframe(values, columns = ['Control', 'Test']
data_frame = data_frame[data_frame['Control']<limit]
data_frame = data_frame[data_frame['Test']<limit]
graphs = sns.jointplot(x = data_frame.Control, y = data_frame.Test, kind = kde, levels = 20)
graphs.ax_joint.plot([2e-4, limit], [2e-4, limit]) 