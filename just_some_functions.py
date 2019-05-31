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

# def probability_of_success(failures, mean_value):
#     '''
#     Using the probability of success for a Poisson distribution.
#     ''' 
#     return mean_value / (failures + mean_value)

def negative_binomial(successes, failures, mu):
    '''
    The number of sucesses.
    The number of failures.
    The mean of the distribution, mu.
    '''

    return (gamma(successes + failures) / (factorial(successes)*gamma(failures))) * ((failures / (mu + failures)) ** failures) * ((mu / (mu + failures)) ** successes)
