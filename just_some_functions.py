import numpy as np
import scipy
import statsmodels.api as sm
import pandas as pd
import seaborn as sns

from math import gamma, lgamma, log
from scipy.stats import beta
from numba import jit



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
    A negative binomial family for when the Poisson distribution 
    has a Gamma distributed mean.

    '''
    def negative_binomial(self, k, y, nu):
        '''
        The number of sucesses.
        The number of failures.
        The mean of the distribution, mu.
        '''

        mu = (k * np.exp(nu)) / (1 - np.exp(nu))

        var = mu * (mu ** 2) / k

        q = mu / (mu + k)

        p = k / (k + mu)

        can_link = log(mu / (k + mu))

        return (gamma(k+y)/(factorial(k)*gamma(y))) * p * (q ** y)

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
    labmdaa = mu ** (2-p)/(phi*(2-p))
    alpha = (2-p)/(1-p)
    gam = phi * (p-1) * (mu ** (p-1))
    N = np.random.poisson(labmdaa, n)
    for i in range(n):
        rt[i] = np.random.gamma(N[i] * np.abs(alpha), gam, 1)
    return rt





# Lemma: calculate probabilities

@jit
def h(a, b, c, d):
    number = lgamma(a + c) + lgamma(a + b) + lgamma(c + d) + lgamma(b + d)
    density = lgamma(a) + lgamma(b) + lgamma(c) + lgamma(d) + lgamma(a + b + c + d)
    return np.exp(number - density)

@jit
def g0(a, b, c):
    return (lgamma(a + c) + lgamma(a + c) - lgamma(a + b + c) + lgamma(a))

@jit
def h_iter(a, b, c, d):
    while d > 1:
        d -= 1
        yield h(a, b, c, d) / d

def g(a, b, c, d):
    return g0(a, b, c) + sum(h_iter(a, b, c, d))

def calculate_probabilities(beta1, beta2):
    return g(beta1.args[0], beta1.args[1], beta2.args[0], beta2.args[1])

def A_B_test(impressions_control_number, conversions_control_number, impressions_test_number, conversions_test_number):

    '''
    relate whatever text to a numerical value in control and test sets 
    worst case scenario can be done via dataframe/index
    '''

    impressions_control, conversions_control = impressions_control_number, conversions_control_number
    impressions_test, conversions_test = impressions_test_number, conversions_test_number

    a_control, b_control = conversions_control, impressions_control - conversions_control + 1
    beta_control = beta(a_control, b_control)
    a_test, b_test = conversions_test, impressions_test - conversions_test + 1
    beta_test = beta(a_test, b_test)

    lift = (beta_test.mean() - beta_control.mean()) / beta_control.mean()
    probability = calculate_probabilities(beta_test, beta_control)
    
    return lift, probability


    # uncomment for visuals
    # values_control = np.random.beta(a_control, b_control, 1e6)
    # values_test = np.random.beta(a_control, b_control, 1e6)
    # values = np.vstack([values_control, values_test]).T

    # limit = 4e-4

    # data_frame = pd.Dataframe(values, columns = ['Control', 'Test'])
    # data_frame = data_frame[data_frame['Control']<limit]
    # data_frame = data_frame[data_frame['Test']<limit]
    # graphs = sns.jointplot(x = data_frame.Control, y = data_frame.Test, kind = 'kde', levels = 20)
    # graphs.ax_joint.plot([2e-4, limit], [2e-4, limit]) 