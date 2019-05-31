import numpy as np
import scipy
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
    The negative binomial family can be thought of as having
    a mean distributed according to a Gamma distribution.

    '''
    pass


def negative_binomial(successes, failures, mu):
    '''
    The number of sucesses.
    The number of failures.
    The mean of the distribution, mu.
    '''

    p = failures / (mu + failures)
    q = mu / (mu + failures)

    return (gamma(successes + failures) / (factorial(successes)*gamma(failures)) * (p) ** failures * (q) ** successes)
