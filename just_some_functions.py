import numpy as np
import scipy
from math import gamma
import statsmodels.api as sm






class NegativeBinomial(Poisson, Gamma):
    '''
    A negative binomial family when the Poisson distribution 
    has a Gamma distributed mean.

    '''
    def negative_binomial(self, k, r, mu):
        '''
        The number of sucesses.
        The number of failures.
        The mean of the distribution, mu.
        '''

        lambdaa = mu ** (2-p)
        p = r / (mu + r)
        q = mu / (mu + r)
        return (gamma(k+r)/(factorial(k)*gamma(r))) * (p**r) * (q**k)


    def factorial(n):
        if n < 0 and n%2 == 0:
            print("Please enter a non-negative integer.")
        elif n < 0 and n%2 != 0:
            print("Please enter a non-negative integer.")
        elif n == 0 or n == 1:
            return 1
        return n * factorial(n-1)


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
    
